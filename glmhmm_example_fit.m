% example code for fitting a glm-hmm on a simulated towers task data
% additionally checks parameter recovery

%% first, simulate some data
% we'll assume that a glm-hmm generated the data
n_sess = 30;                          % number of sessions to simulate
trial_min = 200;                      % minimum number of trials in a session
trial_max = 500;                      % maximum number of trials in a session
f = @(w,x) 1./(1+exp(-w'*x));         % function handle for logistic sigmoid
nz = 3;                               % number of latent states
% poisson rates for "high" and "low" towers sides
mean_high = 6.4;
mean_low = 1.3;

% we're going to assume a design matrix with the following behavioral
% variables:
% x = [bias; last_choice; last_reward; contrast(left-right)]
x_labels = {'bias', 'last choice', 'last reward', 'contrast'};
x_inc = logical([1 1 1 1]);              % features to include
nx = sum(x_inc);                         % number of features in design matrix

% stuff to look at for parameter recovery
n_sims = 10;                                     % number of simulations
nstarts = 10;                                    % number of initializations to find best model fit
model_sim = cell(1,n_sims);                      % store simulated models
model_fit = cell(1,n_sims);                      % store fit models
[ws_sim,ws_fit] = deal(nan(nx,n_sims,nz));       % store GLM weights
[As_sim,As_fit] = deal(nan(nz*nz,n_sims));       % store transition matrices
[pis_sim,pis_fit] = deal(nan(nz,n_sims));        % store initial state probs
[lls_sim,lls_fit] = deal(nan(1,n_sims));         % store log-likelihoods

% simulate session data
for sim_i = 1:n_sims
    fprintf(['sim ',num2str(sim_i),'\n']);
    % generate session lengths
    n_trials = randi([trial_min trial_max],1,n_sess); % random number of trials per session between defined bounds
    nTrials = sum(n_trials);                          % total number of trials across all sessions
    py_z_all = nan(1,nTrials);
    new_sess = false(1,nTrials);                      % logical array indicating the start of each session
    sess_start = 1;
    for i = 1:n_sess
        new_sess(sess_start) = true;
        sess_start = sess_start + n_trials(i);
    end
    
    confusable = 1;
    % since the weights are randomly drawn, any pair of latent states may
    % be too similar in how they predict choices. this causes the recovered
    % fit to "confuse" the two states in weird ways. I'm using this
    % variable (defined below) as a crude check to make sure the simulated
    % states are separable
    while any(confusable(:)>0.5) % this thresholds were chosen non-rigorously by trial-and-error
        
        % draw random GLM weights
        w_sim = normrnd(0,0.5,nx,nz);
        big_w = randsample(1:nx,nz,false);
        for zi = 1:nz
            w_sim(big_w(zi),zi) = normrnd(0,2);
        end
        
        % draw random, weighted transition matrix using a beta distribution
        % this causes the diagonal to be biased towards 1 for "sticky"
        % states
        rs = betarnd(30,2,1,nz); % probability of the diagonal
        A_sim = zeros(nz);
        for zi = 1:nz
            non_z = setdiff(1:nz,zi);
            tmp = rand(1,nz-1); % randomly distribute remaining probability to remaining state transtions
            off_rs = tmp/sum(tmp) * (1-rs(zi));
            A_sim(zi,non_z) = off_rs;
            A_sim(zi,zi) = rs(zi);
        end
        if nz == 1, A_sim = 1; end
        
        % randomly draw initial state probabilities
        pi_sim = rand(nz,1);
        pi_sim = pi_sim/sum(pi_sim);
        pz_z = nan(nz,nTrials);
        pz_z(:,1) = pi_sim;
        
        % compute contrast to be used for each trial
        contrast = nan(1,nTrials);
        for trial_i = 1:nTrials
            % get number of stimuli for each side
            if rand > 0.5 % left high
                stim_left = poissrnd(mean_high);
                stim_right = poissrnd(mean_low);
            else % right high
                stim_right = poissrnd(mean_high);
                stim_left = poissrnd(mean_low);
            end
            contrast(trial_i) = stim_left - stim_right;
        end
        
        % behavioral variables to store
        choice = zeros(1,nTrials);
        reward = zeros(1,nTrials);
        
        % populate behavioral data
        x_sim = nan(nx,nTrials);
        for trial_i = 1:nTrials
            
            % behavioral input for the choice
            if new_sess(trial_i)
                x_tmp = [1; 0; 0; contrast(trial_i)]; % last choice and last reward both 0 at the beginning of the session
                pz_z(:,trial_i) = pi_sim; % reset initial state probability
            else
                x_tmp = [1; choice(trial_i-1); reward(trial_i-1); contrast(trial_i)];
            end
            x_sim(:,trial_i) = x_tmp(x_inc);
            
            % determine latent state and choice probability
            zi = find(histcounts(rand,[0; cumsum(pz_z(:,trial_i))]));
            if nz == 1, zi = 1; end
            
            py_z = f(w_sim(:,zi),x_sim(:,trial_i));
            py_z_all(trial_i) = py_z;
            pz_z(:,trial_i+1) = A_sim(zi,:);
            
            % determince choice and reward
            % we're coding choice as [left,right] = [1,-1]
            % we're also coding reward to incorporate a side interaction such that
            % [left reward, right reward, omission] = [1, -1, 0]
            if rand <= py_z % made a left choice
                choice(trial_i) = 1;
                if contrast(trial_i) > 0 
                    reward(trial_i) = 1;
                else
                    reward(trial_i) = 0;
                end
            else % made a right choice
                choice(trial_i) = -1;
                if contrast(trial_i) < 0
                    reward(trial_i) = -1;
                else
                    reward(trial_i) = 0;
                end
            end
        end
        
        % store simulated "model"
        model_sim{sim_i}.w = w_sim;
        model_sim{sim_i}.A = A_sim;
        model_sim{sim_i}.pi = pi_sim;
        
        % compute latent state probabilities and log-likelihood
        y = choice>0; % obervations need to be binary
        [gammas_sim,~,ll_sim] = runBaumWelch(y,x_sim,model_sim{sim_i},new_sess);
        
        % compute choice probability from emissions model
        % this is where we'll check how "confusable" the states are
        tmpy = 1./(1+exp(-model_sim{sim_i}.w'*x_sim));
        py_z = y.*tmpy + (1-y).*(1-tmpy);
        % use Pearson's correlation coefficient between choice
        % probabilities predicted by each latent state to measure how
        % similar they are
        pyz_conf = corrcoef(py_z');
        % subtract out identity matrix; we don't care about within-state
        % correlations
        confusable = pyz_conf - eye(size(pyz_conf));
    end
    
    % stuff for parameter comparison
    lls_sim(sim_i) = ll_sim;
    ws_sim(:,sim_i,:) = w_sim;
    As_sim(:,sim_i) = A_sim(:);
    pis_sim(:,sim_i) = pi_sim;
    
    %% fitting the model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % these are the variables you'll need to fit the model, whether from
    % simulated data or real data
    
    % 1. the observations
    % in our generated behavioral data, we recorded choice as [left,right] =
    % [1,-1]; however, since we're using the logistic sigmoid as our emissions
    % model, we need the observations to be binary. so we'll redefine our
    % choice observations "y" to be [left,right]=[1,0]
    y = choice>0;
    
    % 2. the design matrix
    % we'll use the same format that we used to generated the data
    % x = [bias last_choice last_reward contrast(left-right)]
    % NOTE: this should be the exact same as x_sim above. I've included it
    % here as a reminder of computing it as an input with real behavioral
    % data
    x = nan(length(x_inc),nTrials);
    for trial_i = 1:nTrials
        if new_sess(trial_i)
            x(:,trial_i) = [1; 0; 0; contrast(trial_i)];
        else
            x(:,trial_i) = [1; choice(trial_i-1); reward(trial_i-1); contrast(trial_i)];
        end
    end
    x = x(x_inc,:);
    
    % since gradient descent is only guaranteed to find a local minimum, it
    % can be helpful to run the model with multiple initializations to
    % estimate the global maximum

    model_tmp = cell(1,nstarts);
    ll_tmp = nan(1,nstarts);
    for start_i = 1:nstarts
    %z_match_sim = ones(nz,1);
    %while ((length(unique(z_match_sim)) < nz) || (length(unique(z_match_rec)) < nz)) && start_i < max_starts
        fprintf(['start ',num2str(start_i),'\n']);
        % 3. initial weight matrix
        % we'll randomly draw these again
        w0 = normrnd(0,1,nx,nz);
        
        % 4. (optional) initial transition matrix
        % fitGlmHmm will initialize A to a uniform distribution if no arguments (or
        % an empty argument) is given
        % instead, we'll randomly draw is as we did above
        rs = betarnd(20,2,1,nz);
        A0 = nan(nz);
        for zi = 1:nz
            A0(zi,:) = (1-rs(zi))/(nz-1);
            A0(zi,zi) = rs(zi);
        end
        
        % 5. (optional) logical array indicating the start of sessions
        % oftentimes you'll pool together data from multiple sessions.
        % in order to reinitialize state probabilities (and not treat
        % trials as continuous across sessions), you can provide an array
        % with a logical value for each trial that indicates whether or not
        % that trial was at the start of a session. we defined this above
        % already as "new_sess"
        new_sess;
        
        % 6. (optional) l2 penalty
        % if weights are fitting to be unrealistically large, you can opt
        % to use an l2 penalty that penalizes large weights. this penalty
        % assumes that weights are draw from a normal distribution with a
        % given standard deviation, such thats weights on the tails of
        % this distribution is less likely to occur (i.e. penalizes the log
        % likelihood)
        l2_penalty = true;
        % if set to "true", you'll need to provide an array "theta"
        % containing the standard deviations. you'll need one for each
        % feature in the design matrix
        theta_full = [2 2 2 2]; % we already know that we drew weights from a normal dist. with s.d. 1
        theta = theta_full(x_inc);
        
        % fitting the model
        model_tmp{start_i} = fitGlmHmm(y,x,w0,A0,'new_sess',new_sess,'tol',1e-6,'l2_penalty',true,'theta',theta);
        [~,~,ll_tmp(start_i)] = runBaumWelch(y,x,model_tmp{start_i},new_sess);

    end
    % the estimated "global" maximum should have the greatest likelihood
    [~,best_fit] = max(ll_tmp);
    model = model_tmp{best_fit};
    ll_fit = ll_tmp(best_fit);
    gammas_fit = runBaumWelch(y,x,model,new_sess);
    
    % match recovered states to simulated states by comparing state
    % occupancies
    gammas_corr = corr(gammas_sim',gammas_fit');
    [~,z_match_sim] = max(gammas_corr);
    [~,z_match_rec] = max(gammas_corr,[],2);
    
    % if the best fit isn't finding a 1:1 state match, it might not have
    % found the actual minimum (i.e. more initializations might be needed
    % to find the minimum), or the states aren't sufficiently separable
    if length(unique(z_match_sim)) < nz, keyboard; end
    if length(unique(z_match_rec)) < nz, keyboard; end
    
    % store stuff for parameter comparison
    model_fit{sim_i} = model;
    ws_fit(:,sim_i,z_match_sim) = model.w;
    A_tmp = model.A(z_match_rec,z_match_rec);
    As_fit(:,sim_i) = A_tmp(:);
    pis_fit(:,sim_i) = model.pi(z_match_rec);
    lls_fit(sim_i) = ll_fit;
    
    
end

%% plot recovered vs simulated parameters
x_lab_use = x_labels(x_inc);

% GLM weights
figure;
for xi = 1:nx
    subplot(1,nx,xi);
    plot(ws_sim(xi,:),ws_fit(xi,:),'.');
    rec_corr = xcorr(ws_sim(xi,:),ws_fit(xi,:),0,'normalized');
    title([x_lab_use{xi} '; r = ' num2str(rec_corr)])
    hold on;
    plot([-10 10],[-10 10],'k');
    %axis equal
    axis([-2 2 -2 2])
    ylabel('recovered')
    xlabel('simulated')
end

% transition matrix
figure;
plot(As_sim(:),As_fit(:),'.');
rec_corr = corr(As_sim(:),As_fit(:));
title(['transition matrix; r = ' num2str(rec_corr)])
hold on;
plot([0 1],[0 1],'k');
%axis equal
axis([0 1 0 1])
ylabel('recovered')
xlabel('simulated')

% initial state probability
figure;
plot(pis_sim(:),pis_fit(:),'.');
rec_corr = corr(pis_sim(:),pis_fit(:));
title(['initial state prob; r = ' num2str(rec_corr)])
hold on;
plot([0 1],[0 1],'k');
%axis equal
axis([0 1 0 1])
ylabel('recovered')
xlabel('simulated')

% log-likelihood
figure;
rec_corr = corr(lls_sim(:),lls_fit(:));
plot(lls_sim,lls_fit,'.');
title(['log-likelihood; r = ' num2str(rec_corr)])
ylabel('recovered')
xlabel('simulated')
