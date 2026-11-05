% Get observation and action specifications from the environment
env = tpgf_11();
numAgents=11;
for i = 1:numAgents
    % Check observation dimension
    obsDim = env.getObservationInfo{i}.Dimension;

    % Get the number of possible actions for this agent
    numActions = numel(env.getActionInfo{i}.Elements);

    % Define network layers with correct input size and output size
     networkLayers = [
     featureInputLayer(obsDim(1), "Normalization", "none", "Name", "state")   
     %sequenceInputLayer([obsDim 1 1], "Normalization", "none", "Name", "state") % Match observation dimension
     %lstmLayer(100, 'OutputMode', 'sequence', 'Name', 'LSTM1') % Add the LSTM layer here
     %lstmLayer(100, 'OutputMode', 'sequence', 'Name', 'LSTM2') % Add the LSTM layer here
       % lstmLayer(100, 'OutputMode', 'sequence', 'Name', 'LSTM3') % Add the LSTM layer here

   fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'fc2') % Second hidden layer with 128 units
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(64, 'Name', 'fc3') % Second hidden layer with 128 units
    reluLayer('Name', 'relu3')
        fullyConnectedLayer(numActions, "Name", "output") % Match number of actions
    ];

    criticOpts = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1,'Optimizer',"Adam",'UseDevice',"gpu");

    % Define critic using the corrected networkLayers
    critic = rlQValueRepresentation(networkLayers, env.getObservationInfo{i}, env.getActionInfo{i}, "Observation", "state", criticOpts);



    % Define agent options and create agent
    agentOptions = rlDQNAgentOptions( ...
        "UseDoubleDQN", true, ...
        "TargetSmoothFactor", 1e-2, ...
        "ExperienceBufferLength", 1e6, ...
        "MiniBatchSize", 64, ...
        "DiscountFactor", 0.95, ...
        'SequenceLength', 1....
        );



        agentOptions.EpsilonGreedyExploration.EpsilonDecay = 1e-3;



    % Create and store the DQN agent
    agents{i} = rlDQNAgent(critic, agentOptions);
end

% Set up the training options
trainOpts = rlMultiAgentTrainingOptions( ...
    "MaxEpisodes", 310, ...
    "MaxStepsPerEpisode", 100, ...
   'StopOnError','on',...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',Inf,...
    ScoreAveragingWindowLength=10....
    );


% Train the agents within the environment
trainingStats = train([agents{1},agents{2},agents{3},agents{4},agents{5},agents{6},agents{7},agents{8},agents{9},agents{10},agents{11}],env,trainOpts);
%trainingStats = train([agents{1},agents{2},agents{3},agents{4},agents{5},agents{6}],env,trainOpts);


save( 'agent30.mat','agents')
