function [model,ll,iter,gammas,ll_norm] = fitGlmHmm(y,x,w0,A0,varargin)
%[model, ll, iter] = fitGlmHmm(y, x, w0, A0, varargin)
% Fits a GLM-HMM to data for a logistic sigmoid emissions models. Number of
% latent states is implicitly taken from the second dimension of w0.
%
% Implementation follows Bishop's "Pattern Recognition and Machine 
% Learning" and Escola et al. (2011)
%
% Inputs:
%   Required:
%     y             - (1 x NTrials) observations
%     x             - (NFeatures x NTrials) design matrix
%     w0            - (NFeatures x NStates) initial latent state 
%                     GLM weights
%
%   Optional:
%     A0            - (NStates x NStates) initial transition matrix. If
%                     unspecified, set to a uniform distribution.
%     maxiter       - (integer, default=1000) number of EM iterations
%     tol           - (float, default=1E-5) difference/"tolerance" in
%                     log-likelihood to determine convergence
%     new_sess      - (1 x NTrials, default = []) logical array with 1s
%                     denoting the start of a new session. If empty, treats
%                     the full set of trials as a single session.
%     l2_penalty    - (logical, default = false) parameter to specify
%                     whether or not to use an l2 penalty for GLM weights
%     theta         - (1 x NFeatures) standard deviations for each GLM
%                     weight if using an l2 penalty
%
% Outputs:
%   model           - GLM-HMM model parameters
%    .w             - (NFeatures x Nstates) latent state GLM weights
%    .w_hess        - Hessian of w 
%    .pi            - (NStates x 1) initial latent state probability
%    .A             - (NStates x NStates) latent state transition matrix
%   ll              - (1 x NIter) log-likelihood of the model fit
%   iter            - number of iterations it took for the model to
%                     converge
%   gammas          - (NStates x NTrials) marginal posterior distribution 
%                     from final fit (from runBaumWelch)
%   ll_norm         - (float) normalized log-likelihood of final fit
%                     computed as ll_norm = exp(ll(end)/NTrials) (from
%                     runBaumWelch)
%
% Example call:
% [model,ll,iter] = fitGlmHmm(y,x,w0,A0,'new_sess',new_sess,'maxiter',2000)

%% %%%%%%%%%%%%%%%%%%%%%% Input parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip = inputParser;
ip.addParameter('maxiter', 10000);      % maximum number of iterations
ip.addParameter('tol', 1E-5);          % difference in log-likelihood to meet completion criteria
ip.addParameter('new_sess', []);       % to reinitialize state probs with new session
ip.addParameter('l2_penalty', false);  % indicates whether or not to use an L2 penalty for MAP
ip.addParameter('theta', []);          % standard deviations of the feature priors for MAP
ip.parse(varargin{:});
for i = fields(ip.Results)', p.(i{1}) = ip.Results.(i{1}); end


%% %%%%%%%%%%%%%%%%%%%%%%%%% Run model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize variables
%f = @(w,x) 1./(1+exp(-w'*x));        % function handle for logistic sigmoid
nstates = size(w0,2);                 % number of latent states
ll = nan(1,p.maxiter);                % log-likelihood at the end of each iteration
dLL = 1;                              % initialize LL difference to run loop
i = 0;                                % loop iterator
nf = size(w0,1);                      % number of features
nz = size(w0,2);                      % number of latent states
nt = length(y);                       % number of trials


if p.l2_penalty && isempty(p.theta)
    error('Need vector of standard deviations for L2 penalty');
end

if rank(x) < nf
    warning('Feature matrix x is not full rank, fit may be inaccurate');
end

if isempty(p.new_sess)
    p.new_sess = false(size(y));
    p.new_sess(1) = true;
end

% initialize model
model.pi = ones(nstates,1)/nstates;            % uniform initial state probabilites
if ~exist('A0','var') || isempty(A0)
    A0 = ones(nstates)./sum(ones(nstates));    % uniform transition probabilities
end
% another way to initialize the transition matrix is to use a beta
% distribution, since HMMs like sticky states
% rs = betarnd(30,2,1,nstates);
% A0 = nan(nstates);
% for z = 1:nstates
%     A0(z,:) = (1-rs(z))/(nstates-1);
%     A0(z,z) = rs(z);
% end

model.A = A0;
model.w = w0;

strcr = []; % string to print progress
while i<p.maxiter && (dLL>p.tol || dLL<0)
  i = i+1;
  
  %% %%%%%%%%%%%%%%%%%%% E-Step: Baum-Welch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  [gammas, xis, ll(i)] = runBaumWelch(y,x,model,p.new_sess);
  if i>1, dLL = ll(i) - ll(i-1); end      % compute difference if >1 iterations have occurred
  
  %% %%%%%%%%%%%%%%% M-Step: Update model parameters %%%%%%%%%%%%%%%%%%%%%%

  % initial state probability - eq. 13.18
  % calculated only from states at the start of a session
  tmpPi = mean(gammas(:,p.new_sess),2);
  model.pi = tmpPi/sum(tmpPi);
  
  % transition matrix - eq 13.19
  %   tmpA = squeeze(sum(xis,3));
  %   model.A = tmpA./sum(tmpA,2);  % normalized across rows (2nd paragraph of 13.2)
  % this is assuming an already summed xi (see NOTE in runBaumWelch.m)
  model.A = xis./sum(xis,2);


  % minimize negative log-likelihood to update glm weights
  func2min = @(wm) negloglik(wm,y,x,gammas,nf,nz,nt,p.l2_penalty,p.theta);
  [w_new,~,~,~,~,w_hess] = fminunc(func2min,model.w,optimoptions('fminunc','Display','off','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective'));
  %[w_new] = fminunc(func2min,model.w,optimoptions('fminunc','Display','off'));%,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective'));

  model.w = w_new;
  model.w_hess = w_hess; % the hessian can be used as an error estimation of w
  
  % string to print progress
  strout =['iteration: ' num2str(i) '; dLL = ' num2str(dLL)];
  fprintf([strcr strout]);
  strcr = repmat('\b',1,length(strout));
  
end
[gammas, ~, ll(i+1),ll_norm] = runBaumWelch(y,x,model,p.new_sess);
ll(isnan(ll)) = []; % get rid of nan values
iter = i;
fprintf(' done\n');

end

function [f,g,h] = negloglik(wm,y,x,gammas,nf,nz,nt,l2_penalty,theta)
% the log-likelihood function is given by the relevant term from eq 2.36
% in Escola et al., Neural Computation, 2011. 

% Bishop eq. 4.90 for negative log-likelihood of logistic sigmoid
%f = -sum(sum(gammas .* (y.*(-log(1+exp(-wm'*x))) + (1-y).*(-wm'*x - log(1+exp(-wm'*x)))),1),2);
pyexp = 1+exp(-wm'*x);
f = -sum(sum(gammas .* (y.*(-log(pyexp)) + (1-y).*(-wm'*x - log(pyexp))),1),2);

% Bishop eq. 4.91, 4.96 for gradient
py = 1./pyexp;
g = x*(gammas.*(py-y))';

% L2 penalty for large weights
if l2_penalty
    l2f = 0.5 * wm'/sparse(1:nf,1:nf,theta)*wm;  % L2 penalty/log-likelihood of the prior
    f = f+sum(l2f(:)); 
    l2g = sparse(1:nf,1:nf,theta)\wm;    % gradient of the L2 penalty
    g = g+l2g;
end

% Bishop eq. 4.97 for hessian
h = zeros(nf*nz);
R = (gammas.*py.*(1-py));
for z = 1:nz
    zind = (z-1)*nf+1;
    
    if l2_penalty
        % add hessian of L2 penalty
        h(zind:(zind+nf-1),zind:(zind+nf-1)) = x*sparse(1:nt,1:nt,(R(z,:)))*x' + inv(sparse(1:nf,1:nf,theta)); 
    else
        h(zind:(zind+nf-1),zind:(zind+nf-1)) = x*sparse(1:nt,1:nt,(R(z,:)))*x';
    end
end

end

