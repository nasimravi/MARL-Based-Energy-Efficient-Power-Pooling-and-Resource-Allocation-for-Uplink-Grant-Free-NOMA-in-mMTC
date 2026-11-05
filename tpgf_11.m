classdef tpgf_11 < rl.env.MultiAgentEnvironment 
  
properties
    n=1001
    packets=zeros(1,1001)
    dis
    energyData = []
    %seed=0 
    energyDataall
    success_prbabilityData = []
    energyall = []
    success = []
    num_active_all = []
end 

methods

    function this = tpgf_11()
    ObservationInfo = { rlNumericSpec([14 1]), rlNumericSpec([15 1]), rlNumericSpec([15 1]),... 
                      rlNumericSpec([15 1]), rlNumericSpec([15 1]), rlNumericSpec([15 1]),... 
                      rlNumericSpec([15 1]), rlNumericSpec([15 1]), rlNumericSpec([15 1]),... 
                      rlNumericSpec([15 1]), rlNumericSpec([15 1])};
     ActionInfo = {rlFiniteSetSpec(1:4),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1]),rlFiniteSetSpec([0 1])};

     this = this@rl.env.MultiAgentEnvironment(ObservationInfo,ActionInfo);

end


function [InitialObservation, InitialState]  = reset(this)
    this.packets=zeros(1,1001);
     InitialState.State = {[0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          };
    InitialObservation=InitialState.State;

    %for simulation
    this.energyDataall= [this.energyDataall sum(this.energyData)];
    
    total_devices = 1001; % Total number of devices

    %generating devices in each region uniformly
    regions = 5; % Number of regions
    region_bounds = linspace(0, 500, regions+1); % Define region boundaries
    
    % Pre-allocate array to store distances
    this.dis = zeros(1, total_devices);
    
    % Assign each device to a random distance within the defined regions
    for i = 1:total_devices
        % Randomly select a region, with probability balanced by region count
        region_idx = randi(regions); 
    
        % Assign a random distance within the selected region
        this.dis(i) = region_bounds(region_idx) + ...
                              rand * (region_bounds(region_idx+1) - region_bounds(region_idx));
    end
    
    %this.seed=0;

end



function [obs , reward, isDone] = step(this, action)
    %% Parameters
    noise_dbm = -120;
    noise = 10^(noise_dbm/10);
    gamma_db = 0;
    gamma = 10^(gamma_db/10);
    alpha = 4;      % path-loss exponent
    ms=10^-3;
    Slot = 1*ms;  %slot Time_s length
    N_RAR = 40; %RAR window size 1 2 4 8 10 20 40 80 slots
    Pr=80*(10^-3);  %receive power
    rngs = [0, 100, 200 , 300 , 400 , 500];
    %base energy to not zero in the denominator
    Energy =1e-140;
%% Generatings
    % this.seed=this.seed+1;
    % rng(this.seed)
    arivval=rand;
    ir = 0 + arivval * 0.01;  % Scale arivval to be between 0 and 0.5
    device_status= zeros(1,this.n);
    device_status(this.packets>0)=-1;
    new_packets= random('poisson',ir,1,1001);
    num_new_packet= length(find(new_packets>0));
    this.packets=this.packets+new_packets;
    active_devices= find(this.packets>0);
    num_active= length(find(this.packets>0));
    collisioned_CTU=0;
    unused_CTU=0;
    channel=zeros(1,this.n);
    channel_gain = exprnd(1,1,this.n);   %generate channel gain
    
    
    tp=action{1};
    td=(5-tp)*8;
    xi = 64*tp; 
    preambles = randi(xi,[1,this.n]);    %each UE randomly selects the preamble
    SINR = zeros(1,this.n);              %initiate SINR at the beginning of each Time_s slot
    preambles(find(this.packets==0)) = 0;                    %clear the preambles of UEs which are not active
    channel_fading = exprnd(1,1,1001);
    channel_gain=channel_fading.*(this.dis.^-alpha);
    channel_gain(find(this.packets==0))=0;
    next_channel_rr = 1; % Start round robin at channel 1
    [~, sorted_indices] = sort(this.dis(active_devices));
    sorted_active_devices = active_devices(sorted_indices);
    for device=sorted_active_devices
        if device_status(device)==-1
            % Device failed in the previous slot -> round-robin channel selection
            channel(device) = next_channel_rr;
            next_channel_rr = mod(next_channel_rr, td) + 1;
        else
            channel(device) = randi(td);
        end
        
    end
    channel(find(this.packets==0)) = 0;
    actionall=[action{2}, action{3}, action{4}, action{5}, action{6}, action{7}, action{8}, action{9}, action{10}, action{11}];
    pp_total = [action{2}, action{3}, action{4}, action{5}, action{6}, action{7}, action{8}, action{9}, action{10}, action{11}]  .*[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
    %pp_total = [1 1 1 1 1 1 1 1 1 1]  .*[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];  
   % pp_total=[1];
   % 
   % 
   %  array = zeros(1, 10);
   %  % 
   %  % Randomly select the number of elements to set to 1 (between 0 and 10)
   %  num_ones = randi([0, 10]);
   % 
   %  % Randomly choose indices to set to 1
   %  one_indices = randperm(10, num_ones);
   % 
   %  % Set the selected indices to 1
   %  array(one_indices) = 1;
   %      pp_total = array  .*[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];


            % des=find(this.dis>= 0 & this.dis< 100);
            % power_level(des)=0.1;
            % des=find(this.dis>= 100 & this.dis< 200);
            % power_level(des)=0.4;
            % des=find(this.dis>= 200 & this.dis< 300);
            % power_level(des)=0.6;
            % des=find(this.dis>= 300 & this.dis< 400);
            % power_level(des)=0.8;
            % des=find(this.dis>= 400 & this.dis< 500);
            % power_level(des)=1;





    pp=pp_total(pp_total ~= 0);
     if all(pp == 0)
         reward=[-10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10];

         obs = {[0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          [0;0;0;0;0;0;0;0;0;0;0;0;0;0;0],...
                          };
         isDone=false;

     else
        random_indices = randi(numel(pp), 1, this.n);
        
        
        for i = 1:length(active_devices)
            if length(pp)>1
                first_half_pool = pp(1:ceil(length(pp)/2));
                second_half_pool = pp(ceil(length(pp)/2)+1:end);
        
        
                
                    %this.seed=this.seed+2;
                    if this.dis(active_devices(i))<250
                        power_level(active_devices(i))= randsample(first_half_pool, 1);
                    else
        
                        power_level(active_devices(i))= randsample(second_half_pool, 1);
        
                    end
            else
                power_level(active_devices(i))=pp;
            end

        end
         %for i=1:7
         % if length(pp)>1
         %    des=find(this.dis>= 0 & this.dis< 250);
         %    power_level(des)=randsample(pp(1:ceil(length(pp)/2)), 1);
         %    des=find(this.dis>= 250 & this.dis< 500);
         %    power_level(des)=randsample(pp(ceil(length(pp)/2)+1:end), 1);
         % end
            % des=find(this.dis>= 400 & this.dis< 600);
            % power_level(des)=randsample([0.4, 0.5, 0.6], 1);
            % des=find(this.dis>= 600 & this.dis< 700);
            % power_level(des)=randsample([0.6, 0.7, 0.8], 1);
        %end



       % power_level = pp(random_indices);
        power_level(find(this.packets==0))=0;
        power_r= power_level.*(channel_gain);
        power_dbm = 10 * log10(power_r * 1000);
        power_db = 10.^(power_dbm/10);
    
        Energy =Energy+(sum(power_level(find(this.packets>0)))*2*Slot);
        eng = zeros(1,10);
        for i=1:10
            powerlevel_value = i/10;
            
            % Calculate the energy for devices selecting the current power level
            eng(i) = 0.0000000001+sum(power_level((this.packets > 0) & (power_level == powerlevel_value))) * 2 * Slot;
        end
        engall=[eng(1), eng(2), eng(3), eng(4), eng(5), eng(6), eng(7), eng(8), eng(9), eng(10)];


            % number of actives in each power level
                for i=1:10
            powerlevel_value = i/10;
            
            active(i) = sum((this.packets > 0) & (power_level == powerlevel_value));
                end

                active_all=[active(1), active(2), active(3), active(4), active(5), active(6), active(7), active(8), active(9), active(10)];

                active_half(1)=length(find(this.dis(this.packets>0)<250));
                active_half(2)=length(find(this.dis(this.packets>0)>250));
                


        Time_s =(2*Slot);
        %baraye inke bebinim unayike ferestade success bude ya na!
        for xi_det = 1:xi                       %detect each preamble
            Index = find(preambles==xi_det);    %find the UEs select the same preamble
            if length(Index)>1
                 preambles(find(preambles==xi_det))=0;   %clear preamble xi_det
                 Energy = Energy+ length(Index)*N_RAR*Pr*Slot; %calculates the No RAR energy part
                 Time_s =Time_s+length(Index)*N_RAR*Slot;
                 
    
            end
        end
        
        [sorted_channelgain, sorted_indices_ch] = sort(channel_gain,'descend'); %for channel gain
        
            % Find indices of non-zero elements  
        preamble_success = find(preambles>0);
        num_preamble_success=length(preamble_success);
        preamble_f=num_active-num_preamble_success; %preamble failure;
        channel_powers = cell(1,td);
        channel_indices = cell(1,td);
        channel_r=cell(1,td);
    
        % Group devices based on the channel they select
        for k=1:td
            % Find devices on the current channel
            devices_on_channel = find(channel == k);
            
            % Store power and indices for the current channel
            channel_powers{k} = power_level(devices_on_channel);
            channel_r{k} = power_r(devices_on_channel);
            channel_indices{k} = devices_on_channel;
    
            if devices_on_channel==0
                unused_CTU=unused_CTU+length(pp);
            else
    
                % Count how many devices are using each power level on this channel
                for i = 1:length(pp)
                    if (power_level(devices_on_channel) == power_level(i))==0
                        unused_CTU=unused_CTU+1;
                    elseif length(find(power_level(devices_on_channel) == power_level(i)))>1
                        collisioned_CTU=collisioned_CTU+1;
                    end
        
                end
            end
    
        end
    
    
        for k = 1:td
    
             % Sort power and indices for each channel
            [sorted_power_r, sorted_indices] = sort(channel_r{k}, 'descend');
            sorted_indices = channel_indices{k}(sorted_indices);
            total_devices_on_channel= length(find(channel_r{k}>-inf));
    
           for ch = 1:total_devices_on_channel 
             interference = sum(sorted_power_r(ch+1:total_devices_on_channel));
    
            SINR(sorted_indices(ch)) = power_r(sorted_indices(ch)) / (interference + noise); % calculates SINR for each UE
            if SINR(sorted_indices(ch)) < gamma
                break;
            end
           end
        end
       
        SINR(preambles==0)=0;
        succ_index = find(SINR>=gamma);   %UEs whose Msg3 SINR higher than threshold
        for i=1:10
            powerlevel_value = i/10;
            
            % Calculate the energy for devices selecting the current power level
            suc(i) = sum((SINR>=gamma) & (power_level == powerlevel_value));
        end

        succ_all=[suc(1), suc(2), suc(3), suc(4), suc(5), suc(6), suc(7), suc(8), suc(9), suc(10)];

         for i=1:10
            ef(i)=(suc(i)/eng(i))/200;
         end

         ef_all=[ef(1)/100, ef(2)/100, ef(3)/100, ef(4)/100, ef(5)/100, ef(6)/100, ef(7)/100, ef(8)/100, ef(9)/100, ef(10)/100];

    
        preambles(find(SINR<gamma))=0;              %clear the preambles of UEs whose Msg3 SINR lower than the threshold
        Energy = Energy+ (length(succ_index)*(N_RAR/2)*Slot*Pr)+...    %calculates success Msg2 and Msg3 energy part
       ((length(preamble_success)-length(succ_index))*N_RAR*Pr*Slot);    %calculates success Msg2 and Msg3 energy part       
        Time_s= Time_s+length(succ_index)*((N_RAR/2)*Slot)+...    %calculates success Msg2 and Msg3 energy part
       ((length(preamble_success)-length(succ_index))*N_RAR*Slot);    %calculates success Msg2 and Msg3 energy part       
        this.packets(succ_index) = this.packets(succ_index)-1;            %update buffer
        num_succ = length(succ_index);
       energy_Efficiency= num_succ/Energy;
       success_probability= num_succ/num_active;
       if num_active==0
            success_probability=1;
       end
        faiulre=length(find(this.packets>0));
    
    
        % for i=1:5
        %      fail_dis1= this.dis(find(this.packets>0))>= rngs(i) & this.dis(find(this.packets>0))< rngs(i+1));
        %      fail_dist1(i)=length(fail_dis1);
        % end 

       fail_dis= find(this.packets>0 & this.dis< 250);
       fail_dist(1)=length(fail_dis);
       fail_dis= find(this.packets>0 & this.dis> 250);
       fail_dist(2)=length(fail_dis);
       

       fail_distall=[fail_dist(1), fail_dist(2)];


    
    
       isDone=false;
    if ~isDone
        reward =  [energy_Efficiency+num_preamble_success energy_Efficiency+ef(1) energy_Efficiency+ef(2) energy_Efficiency+ef(3) energy_Efficiency+ef(4) energy_Efficiency+ef(5) energy_Efficiency+ef(6) energy_Efficiency+ef(7) energy_Efficiency+ef(8) energy_Efficiency+ef(9) energy_Efficiency+ef(10)];
    
    else
        reward=0;
    end
 
    
    this.energyData = [this.energyData, energy_Efficiency];
    this.success_prbabilityData = [this.success_prbabilityData, success_probability];
    this.energyall=[this.energyall, Energy];
    this.success=[this.success num_succ];
    this.num_active_all=[this.num_active_all num_new_packet];
    
    obs = {[action{1};action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{11};num_active;num_preamble_success;num_succ],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       [active_half(1);active_half(2);fail_dist(1);fail_dist(2);action{2};action{3};action{4};action{5};action{6};action{7};action{8};action{9};action{10};action{1};action{11}],...
       };

    
    
     end

        end     

    end  

end 