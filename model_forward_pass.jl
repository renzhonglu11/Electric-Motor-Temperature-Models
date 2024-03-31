# module ModelForwardPass  # begin module





module ModelForwardPass  # begin module

# Since this is a separate namespace, the packages included outside will not
# be available (this is to ensure that this can run without your solution
# notebook)
# add all packages that you need for the forward pass below
using StatsBase
using OrdinaryDiffEq, Lux, Random, ComponentArrays
using SciMLSensitivity, Optimization, OptimizationOptimJL, LineSearches, Zygote, OptimizationFlux, LinearAlgebra
using MAT
using Serialization
using DataInterpolations
using Random
using Statistics
using OptimizationOptimisers
using Parameters: @unpack
using Serialization


interp_input = nothing
interp_input_2 = nothing
predict_temp_1 = nothing
i_d_sqr_max, i_q_sqr_max, in_id_sqr_max, in_id_sqr_max = nothing, nothing, nothing, nothing
R_th_guess_stat = 0.4f0; # initial guess for thermal capacitance (off by roughly 50 %)
R_th_guess_rot = 0.07f0;
P_v_guess = 5000f0;




# interpolate inputs for all channels separately
function get_data_continous(u, tsteps)
  
  interp_input = [AkimaInterpolation(u[:, i], tsteps) for i in 1:size(u, 2)]  # deal with more than two features
  return interp_input
end

# interpolation with quadratic spline
function redo_get_data_continous(u, tsteps)
  interp_input = [QuadraticSpline(u[:, i], tsteps) for i in 1:size(u, 2)]  # deal with more than two features
  return interp_input
end

# get data from interpolation with quadratic spline 
function interpolate_input2(t)
    t_float64 = Float64(t)

    return [Float64(DataInterpolations.derivative(f,t_float64)) for f in interp_input_with_last_two]
end


# get data from continued data
function interpolate_input(t, data)
    t_float64 = Float64(t)

    return [Float64(f(t_float64)) for f in data]
end

function get_tspan_tsteps(y, dt)
  NF = Float64
  datasize = size(y,1)
  total_duration = datasize * dt
  tspan = (zero(NF),NF(total_duration))
  tsteps = range(tspan[1], tspan[2]; length=datasize)
  tsteps_vector = collect(tsteps)
  return tspan, tsteps_vector
end


columns_to_keep = unique([16, 17, 18, 19, 20, 21])
columns_to_keep = sort(columns_to_keep, rev=true)

function keep_features(data, features_to_keep)
  # Ensure features_to_keep is sorted and unique to avoid errors or duplications
  features_to_keep = sort(unique(features_to_keep))
  
  # Create a new data array by selecting only the columns in features_to_keep
  new_data = data[:, features_to_keep]
  
  return new_data
end


function remove_feature(data)
  columns_to_remove = unique([2, 3, 7, 11, 12, 13, 14, 15])
  columns_to_remove = sort(columns_to_remove, rev=true)
  new_data = data
  for col in columns_to_remove
    new_data = hcat(new_data[:, 1:col-1], new_data[:, col+1:end])
  end
  return new_data
end

function predict_NODE(w, problem, x0, tsteps, dt)

    new_problem = remake(problem, u0=[x0], p=w)
    
    reduce(vcat, solve(new_problem, Tsit5();saveat = tsteps).u)

end;


NF = Float64


function elu(x, α=1.0)
  return x > 0 ? x : α * (exp(x) - 1)
end

# Define the biased ELU function
function biased_elu(x; α=1.0)
  # Assuming x is a Lux array or compatible structure
  return map(xi -> elu(xi, α) + 1.0, x)
end

# State Prediction Network
state_nn = Lux.Chain(
    Lux.Dense(11, 20, Lux.tanh),  # Input: 21 features + 2 state variables
    # Lux.Dense(20, 20, Lux.tanh),
    Lux.Dense(20, 1, sigmoid)  # Output: 2 state variables (ΔT_stat, ΔT_rot)
)


conductance_net = Lux.Chain(
    Lux.Dense(11, 20, tanh),  
    Lux.Dense(20, 1),  
    biased_elu  
)

p_state, st_state = Lux.setup(MersenneTwister(260787), state_nn)
p_state_conductance, st_state_conductance = Lux.setup(MersenneTwister(260787), conductance_net)


function simple_motor_temperature_model_learn(x, p, t)
   
    @unpack C_th_scale_stat, C_th_scale_rot, R_th_scale, w_ANN, w_conductance = p
    global interp_input, interp_input_2
    global i_d_sqr_max, i_q_sqr_max, in_id_sqr_max, in_iq_sqr_max
    global predict_temp_1
    

   
    i_d = interp_input[4]
    i_q = interp_input[5]
    rmp = interp_input[1]
    oil_temp_entry_rotor = interp_input[18]
    oil_temp_entry_stator  = interp_input[19]
    oil_temp_exit_rotor_a = interp_input[20]
    oil_temp_exit_rotor_b = interp_input[21]
    flow_rate_rotor = interp_input[16]
    flow_rate_stator = interp_input[17]
    
    
    
    n_rad_s = rmp(t) * (2 * π / 60)
    
    
    input_temp = nothing
    if predict_temp_1
        input_temp = x[1] ./ 190.0
    
    else
    
        input_temp = x[1] ./ 140.0
    end
    


    
    ann_input = [
          input_temp
        ; i_d(t).^2 ./ i_d_sqr_max
        ; i_q(t).^2 ./ i_q_sqr_max
        ; (i_d(t) .* n_rad_s).^2 ./ in_id_sqr_max
        ; (i_q(t) .* n_rad_s).^2 ./ in_iq_sqr_max
        ; interpolate_input(t,interp_input_2)] # 3 + 10 = 13  interpolate_input(t,interp_input_2)
    
   
    
    dT_dt = state_nn(ann_input, w_ANN, st_state)  
    C_th_guess = conductance_net(ann_input, w_conductance, st_state_conductance)  
    
    final_dT = nothing
    
    if predict_temp_1 
        approx_ambient_temp_stator = flow_rate_rotor(t) .* 0.1 .* (oil_temp_exit_rotor_a(t) .- oil_temp_entry_rotor(t))
    
        final_dT = (1 / (C_th_scale_rot * C_th_guess[1][1])) * ((-x[1] + approx_ambient_temp_stator) / (R_th_scale * R_th_guess_rot) + P_v_guess * dT_dt[1][1])
    else
        approx_ambient_temp_rotor = oil_temp_exit_rotor_a(t)

        final_dT = (1 / (C_th_scale_stat * C_th_guess[1][1])) * ((-x[1] + approx_ambient_temp_rotor) / (R_th_scale * R_th_guess_stat) + P_v_guess * dT_dt[1][1])
    end
        
    return [final_dT]
end


""" 
    your_model_forward_pass(
        inputs=inputs,
        x0=_x0,
        parameters=parameters_evaluation,
        tsteps=tsteps,
    )

The forward pass for your model that will be used to evaluate its performance.

##### Arguments
- `inputs`: A matrix with shape (N x 21) that contains the N input vector that should be
applied one after the other.
- `x0`: A vector with 2 elements that holds the initial state of the evaluation sequence.
- `parameters`: Any parameters of your model
- `tsteps`: The time steps corresponding to the inputs
- `dt`: The time between two measurements

##### Returns
The state trajectory that your model produced for the given inputs.
""" 
function your_model_forward_pass(;
        inputs,
        x0,
        parameters,
        tsteps,
        dt=0.5
    )
    global interp_input, interp_input_2
    global i_d_sqr_max, i_q_sqr_max, in_id_sqr_max, in_iq_sqr_max
    global predict_temp_1
    
    U_train = inputs
    rpm = U_train[:, 1]
    i_d = U_train[:, 4]  # d-current
    i_q = U_train[:, 5]  # q-current
    i_d_sqr = i_d .^ 2
    i_q_sqr = i_q .^ 2
    n_rad_s = rpm .* (2 * π / 60)

    in_id_sqr = (i_d .* n_rad_s) .^ 2
    in_iq_sqr = (i_q .* n_rad_s) .^ 2 


    i_d_sqr_max = NF(maximum(i_d_sqr))
    i_q_sqr_max = NF(maximum(i_q_sqr))
    in_id_sqr_max = NF(maximum(in_id_sqr))
    in_iq_sqr_max = NF(maximum(in_iq_sqr))    
    
    
    
    # begin placeholder (replace with YOUR CODE)
    x0_1 = x0[1]
    x0_2 = x0[2]
    
    param_1 = parameters[1]
    param_2 = parameters[2]
    
    final_matrix = hcat(inputs)
    
    tspan, tsteps = get_tspan_tsteps(inputs, dt)
    # process features for input u
    
    interp_input = get_data_continous(final_matrix, tsteps) # this dataset has all features
    
    rest_features = keep_features(final_matrix, columns_to_keep) # features that need to be input to ann
    interp_input_2 = get_data_continous(rest_features, tsteps)
    
    
    # ensure to overwrites the global variable training_prob
    predict_temp_1 = true
    training_prob_1 = ODEProblem{false}(simple_motor_temperature_model_learn, x0_1, tspan, param_1)
    prediction_eval_1 = predict_NODE(param_1, training_prob_1, x0_1, tsteps, dt)
    
    predict_temp_1 = false
    training_prob_2 = ODEProblem{false}(simple_motor_temperature_model_learn, x0_2, tspan, param_2)
    prediction_eval_2 = predict_NODE(param_2, training_prob_2, x0_2, tsteps, dt)
    
    trajectory = hcat(prediction_eval_1, prediction_eval_2)
    
    # end placeholder

    return trajectory
end


end  # end module