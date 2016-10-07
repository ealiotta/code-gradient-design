% This fuction generates Convex Optimized Diffusion Encoding (CODE) waveforms
% for a user defined TE subject to gradient hardware constraints, moment
% nulling requirements and sequence timing parameters using the Chambolle-Pock 
% algorithm: 
%
% A First-Order Primal-Dual Algorithm for Convex Problems with Applications 
% to Imaging. Chambolle, A and Pock, T; Journal of Mathematical Imaging and 
% Vision; 2011.
%
% INPUTS  : GMAX            - Max Gradient amplitude [T/m]   
%           SMAX            - Maximum gradient slew rate [T/m/s]
%           Gvec            - Diffusion encoding vector magnitude: sqrt(Gx^2 + Gy^2 + Gz^2)
%                             (for example: encoding diffusion along only x
%                             and y increases the max gradienDt amplitude by
%                             a factor of sqrt(1^2 + 1^2 + 0^2) = 1.414)
%           TE              - Echo Time [ms]
%           TimeToEcho      - EPI time to Echo [ms]
%           TimeToEncode    - Start time of diffusion encoding [ms]
%           RFdur           - Refocusing pulse duration [ms]
%           MMT             - Desired waveform moments [0-M0, 1-M0M1,2-M0M1M2]
%           T               - Timestep of optimization [s] default: 100us
%
% OUTPUTS : G               - CODE gradient waveform
%           TE              - Sequence Echo Time (TE)
%
% Magnetic Resonance Research Labs (http://mrrl.ucla.edu)
% Department of Radiological Sciences
% University of California, Los Angeles
% - Eric Aliotta (EAliotta@mednet.ucla.edu)
% - Holden Wu (HWu@mednet.ucla.edu)
% - Daniel Ennis (DEnnis@mednet.ucla.edu)
% - Dan Nguyen (DanNguyen@mednet.ucla.edu)
% - September 6, 2016

function [G, TE] = CODE_Optimization_CP(G_Max,S_Max,Gvec,bvalue,T_ECHO,T_90,T_RF,MMT)

%% Define default values to generate a test waveform
if nargin==0
    G_Max     = 74e-3;            % T/m
    Gvec      = 1.414;            % magnitude of direction vector (sqrt(Gx^2 + Gy^2 + Gz^2)). For example, Gvec = 1 for only x encoding
    S_Max     = 50;               % T/m/s
    bvalue    = 500;              % target b-value s/mm2
    T_ECHO    = 26.4;             % EPI time to center k-space line [ms]
    T_90      = 5.3;              % Start time of diffusion. Typically the duration of excitation + EPI correction lines [ms]
    T_RF      = 4.3;              % 180 duration. [ms]
    MMT       = 2;                % Desired waveform moments- [0 for M0=0, 1 for M0=M1=0, 2 for M0=M1=M2=0]
end

%% Define some constants

% Simulation constants
dt      = 0.1e-3;              % timestep of optimization [s] (increase for faster simulation)
N_max   = 5e3;                  % Terminate bisection search when n > N_max

% Physical constants
GAMMA   = 42.58e3;            %Hz/mT for protons

%% Define the moment nulling vector
switch MMT
    case 0
        mvec = 0;           % M0 nulled gradients
    case 1
        mvec = [0;0;];      % M0+M1 nulled gradients
    case 2
        mvec = [0;0;0;];    % M0+M1+M2 nulled gradients
end

tA = clock; % start a timer to evaluate optimization speed

shortenLoop = 0;

%% Design the upper-bound symmetric gradient waveforms
[tHigh,G_mono,b_tmp] = design_symmetric_gradients(bvalue,T_ECHO,T_90,G_Max,length(mvec)-1);

%% Define time and index bounds
tLow  = 2*(T_RF/2 + T_ECHO);    % TE of SE (b=0) sequence

n_top = floor(tHigh * 1e-3/dt); % upper bound on TE
n_bot = floor(tLow * 1e-3/dt);  % lower bound on TE

%% Run the optimization
fprintf('Optimizing...... \n');
done = 0; n = n_top;
while( done==0 )
  
fprintf('............ TE <= %2.1fms ... Gap: %2.1f \n' ,n_top*dt/(1e-3), (n_top-n_bot)*dt/(1e-3));

n = floor(n-T_ECHO/(dt*1e3));

tECHO = n + T_ECHO/(dt*1e3);

tINV = floor(tECHO/2);

INV = ones(n,1);   INV(tINV:end) = -1;
C=tril(ones(n));
C2 = C'*C;

t0 = 0;
tf = 0;

D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
D = D(1:end-1,:);

Nm = size(mvec, 1);
tvec = t0 + [0:n-1]*dt; % in sec
% tMat for all moments
tMat = zeros( Nm, n );
for mm=1:Nm,
    tMat( mm, : ) = tvec.^(mm-1);
end

if n > tINV + T_RF/(dt*1e3)/2  
    
TE = n*dt/(1e-3)+T_ECHO;

% call the optimization kernel
grad_tmp = CP_Optim_Kernel(G_Max,S_Max,Gvec,TE,T_ECHO,T_90,T_RF,MMT);

% check b-value of gradient.
b_test = (GAMMA*2*pi)^2*(grad_tmp.*INV*dt)'*(C2*(grad_tmp.*INV*dt))*dt;

% check gradient moments.
mtest = max(abs(GAMMA*dt*tMat*(grad_tmp.*INV)));

if isnan(b_test)
    b_test = 0;
end

if b_test > bvalue
    is_adequate = 'YES';
else
    if abs(b_test-bvalue) <= 0.01*bvalue
        is_adequate = 'YES';
    else
        is_adequate = 'NO';
    end
end

if mtest > 0.01
    is_adequate = 'NO';
end


else
    is_adequate = 'NO';
end

if( strcmp(is_adequate,'YES') ) % feasible
    feas = 1;
    n_top = n+T_ECHO/(dt*1e3);
    n = round( 0.5*(n_top+n_bot) );
    % update feasible g
    G = grad_tmp;
else
    %scatter(1,n,400,'r','x');
    n_bot = n+T_ECHO/(dt*1e3);
    n = round( 0.5*(n_top+n_bot) );
end

% check termination condition
if( n_top<=n_bot+1 )
    if( exist('G') )
        % if the b-value is still too LARGE, scale GMAX down
        if abs(b_test-bvalue) >= 0.1*bvalue
            G = scale_Gmax(G,bvalue,dt,tINV);
            % and try a shorter TE!
            shortenLoop = shortenLoop + 1;
            n_bot = n_bot - 10;
            if shortenLoop == 3
                fprintf('Scaled Down....... DONE \n');
                done = 1;
            end
        else
            fprintf('................ DONE \n');
            done = 1;
        end
        
    else
        % if we're here, n_top was infeasible to start with
        n_top = n_top*2;
        n = n_top;
    end
end
%end
% hard termination condition
if( n > N_max )
    disp( sprintf('mtgrad_cvx: n=%d > N_max=%d, terminating bisection.',n,N_max) );    
    done = 1; 
end

end

close(gcf);

tB = clock;

SimTime = (tB(4)*3600 + tB(5)*60 + tB(6)) - (tA(4)*3600 + tA(5)*60 + tA(6));

if SimTime > 3600
    fprintf('Optimization time: %d hr %d min %2.2f sec \n', floor(SimTime/3600), rem(SimTime,60), SimTime/60 - rem(SimTime,60)*60);
else if SimTime > 60
        fprintf('Optimization time: %d min %2.2f sec \n', floor(SimTime/60), rem(SimTime,60));
    else
        fprintf('Optimization time: %2.2f sec \n', SimTime);
    end
end

n = length(G);

% form difference matrix to calculate slew rate
D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
D = D(1:end-1,:);

C=tril(ones(n));
C2 = C'*C;

INV = ones(n,1);    INV(tINV:end) = -1;

% form time vector to calculate moments
tvec = t0 + [0:n-1]*dt; % in sec
% tMat for all moments
tMat = zeros( 3, n );
for mm=1:3,
    tMat( mm, : ) = tvec.^(mm-1);
end

% progressive tMat for m1, m2
tMat0 = tril(ones(n)).*repmat(tMat(1,:)',[1,n])';
tMat1 = tril(ones(n)).*repmat(tMat(2,:)',[1,n])';
tMat2 = tril(ones(n)).*repmat(tMat(3,:)',[1,n])';

% moment
moments = GAMMA*dt*tMat*(G.*INV);

M0 = GAMMA*dt*tMat0*(G.*INV);
M1 = GAMMA*dt*tMat1*(G.*INV);
M2 = GAMMA*dt*tMat2*(G.*INV);
%

% bvalue
b_val = (GAMMA*2*pi)^2*(G.*INV*dt)'*(C2*(G.*INV*dt))*dt;

tPrep = length(G)*dt/(1e-3);

TE = tPrep + T_ECHO;

% figure; subplot(2,1,1); 
% plot(G,'LineWidth',2);
% legend('G [mT/mm]');
% subplot(2,1,2); 
% plot(moment1/100,'r','LineWidth',2); hold on; plot(moment2,'LineWidth',2);
% fontsizeALL(gcf,20);
% legend('m1','m2');

DESCRIPTION = ['b-value = ' num2str(round(b_val)) ',  TE = ' num2str(TE) '---- Gmax - ' num2str(max(G)/Gvec) ];

fprintf([DESCRIPTION, '\n']);
fprintf('Non-optimized TE = %1.1f\n',tHigh);
fprintf('CODE Benefit:      %1.1f%%\n',(tHigh-TE)*100/tHigh);

%% Generate a figure
figure; subplot(2,1,1);
plot(G,'LineWidth',2);
title(DESCRIPTION); legend('G [mT/mm]');
subplot(2,1,2);
plot(M1/100,'r','LineWidth',2); hold on; plot(M2,'LineWidth',2);
legend('m1','m2');

end


function [TE,G,b] = design_symmetric_gradients(bvalue_target,T_ECHO,T_90,G_Max,MMT)
% Returns the TE for symmetric DWI waveforms with a specified b-value and
% sequence timing parameters. The waveforms used are: MONOPOLAR, BIPOLAR
% and MODIFIED BIPOLAR (Stoeck CT, von Deuster C, Genet M, Atkinson D, Kozerke
%                       S. Second-order motion-compensated spin echo diffusion
%                       tensor imaging of the human heart. MRM. 2015.)
%
% INPUTS:  G_Max        - Max gradient amplitude [T/m]
%          bvalue_target- Desired b-value [s/mm2]
%          T_ECHO       - EPI time to Echo [ms]
%          T_90         - Start time of diffusion encoding [ms]
%          MMT          - Desired waveform moments
%                         - 0 - M0= 0      - MONO
%                         - 1 - M0M1 = 0   - BIPOLAR
%                         - 2 - M0M1M2 = 0 - MODIFIED BIPOLAR
%
% OUTPUTS: TE    -  TE of resultant waveform [ms]
%          G     -  Diffusion encoding waveform [T/m]
%          b     -  b-value of encoding waveform [s/mm2]
%
% Magnetic Resonance Research Labs (http://mrrl.ucla.edu)
% Department of Radiological Sciences
% University of California, Los Angeles
% Eric Aliotta (EAliotta@mednet.ucla.edu)
% Holden Wu (HWu@mednet.ucla.edu)
% Daniel Ennis (DEnnis@mednet.ucla.edu)
% December 16, 2015

epsilon = 1.5;   % gradient ramp time 
RFgap     = 4.3; % 180 pulse duration
epsilon = floor(epsilon*10)/10;

% define monopolar waveform
if MMT == 0
  gap = RFgap;
  N = 4*epsilon + gap + 2*T_ECHO+T_90; % starting point for total duration
  T = 0.1; % scale time in ms
  b = 0;
  
  % update waveform until the b-value is large enough
  while(b<bvalue_target*0.995)
    N = N+T;
    
    time = N;
    
    lambda = (time - 4*epsilon - gap - 2*T_ECHO)/2;
    lambda = round(lambda/T);
    
    grad = trapTransform([lambda,lambda],G_Max,floor(epsilon/T),1,floor((T_ECHO-T_90+gap)/T),1);
    
    n = length(grad);
    
    C=tril(ones(n));
    C2 = C'*C;
    GAMMA = 42580;
    
    INV = ones(n,1);   INV(floor((n+T_ECHO)/2):end) = -1;
    
    Ts = T*(1e-3);
    
    b = (GAMMA*2*pi)^2*(grad.*INV*Ts)'*(C2*(grad.*INV*Ts))*Ts;
    
    tINV = ceil(lambda + floor((T_ECHO-T_90+gap)/T) + 2*epsilon/T - 0.5*gap/T);
    TEh1 = T_ECHO/T + length(grad(tINV:end));
    TEh2 = tINV;
    
    TE = 2*max(TEh1,TEh2)*T;
    G = grad;
  end
end

% define bipolar waveform (M1=0)
if MMT == 1
  
  L = 1; % starting point
  T = 0.1; % scale time in ms
  
  b = 0;
  % update waveform until the b-value is large enough
  while(b<bvalue_target*0.995)
    
    L = L+T;
    
    % duration of each bipolar lobe
    lambda  = L;         
    LAMBDA  = lambda;    
    
    LAMBDA = round(LAMBDA/T);
    lambda = round(lambda/T);

    % gap b/w gradients is just the RF duration 
    gap = RFgap;
    
    % take trapezoid durations and create G(t) vector
    grad = trapTransform([lambda,-LAMBDA,LAMBDA,-lambda],G_Max,round(epsilon/T),1,round(gap/T),2);
    
    % legnth of waveform
    n = length(grad);
    
    % vector for b-value integration
    C=tril(ones(n));
    C2 = C'*C;
    
    % Gyromagnetic ratio
    GAMMA = 42580;
    
    % refocusing pulse time
    tINV = floor(n/2);
    
    % vector to invert magnetization (+1 before 180, -1 after 180)
    INV = ones(n,1);
    INV(floor(tINV):end) = -1;
    
    % time increment in seconds
    Ts = T*(1e-3);
    
    % test b-value
    b = (GAMMA*2*pi)^2*(grad.*INV*Ts)'*(C2*(grad.*INV*Ts))*Ts;
    
    % pre 180 contribution to TE
    TEh1 = 0.5*RFgap/T + lambda + LAMBDA + 4*epsilon/T + T_ECHO/T;
    % post 180 contribution to TE
    TEh2 = 0.5*RFgap/T + lambda + LAMBDA + 4*epsilon/T + T_90/T;
    
    % Final TE
    TE = 2*max(TEh1,TEh2)*T + 2 + 2; %additional 4ms for spoilers.
    
    % final gradient waveform
    G = grad;
  end
end

% define modified bipolar (M1=M2 = 0) waveform
if MMT == 2
  L = 1; % starting point
  T = 0.1; % scale in ms
  
  b = 0;
  
  % update waveform until the b-value is large enough
  while(b<bvalue_target*0.995)
    
    L = L+T;
    
    % first trap duration
    lambda = L;                     lambda = round(lambda/T); 
    % second trap duration
    LAMBDA  = 2*lambda + epsilon;   LAMBDA = round(LAMBDA/T);
    
    % time between first and second sets of gradients
    gap = 2*epsilon + lambda;
    
    % take trapezoid durations and create G(t) vector
    grad = trapTransform([lambda,-LAMBDA,-LAMBDA,lambda],G_Max,round(epsilon/T),1,round(gap/T),2);
    
    % legnth of waveform
    n = length(grad);
    
    % vector for b-value integration
    C=tril(ones(n));
    C2 = C'*C;
    
    % Gyromagnetic ratio
    GAMMA = 42580;
    
    % refocusing pulse time
    tINV = n/2 + round(gap/T) - round(RFgap/T);
    
    % vector to invert magnetization (+1 before 180, -1 after 180)
    INV = ones(n,1);
    INV(floor(tINV):end) = -1;
    
    % time increment in seconds
    Ts = T*(1e-3);
    
    % test b-value
    b = (GAMMA*2*pi)^2*(grad.*INV*Ts)'*(C2*(grad.*INV*Ts))*Ts;
    
    % pre 180 contribution to TE
    TEh1 = 0.5*RFgap/T + lambda + LAMBDA + 4*epsilon/T + T_ECHO/T;
    % post 180 contribution to TE
    TEh2 = -0.5*RFgap/T + lambda + LAMBDA + 4*epsilon/T + T_90/T + gap/T;
    
    % final TE
    TE = 2*max(TEh1,TEh2)*T;
    
    % final gradient waveform
    G = grad;
  end
end


end

%% CODE Chambolle-Pock Optimization Kernel

function G = CP_Optim_Kernel(GMAX,SRMAX,Gvec,TE,TimeToEcho,TimeToEncode,RFdur,MMT,T)

if nargin<9
    T = 0.1e-3; % timestep %s
end

SRMAX = SRMAX*T*0.5*Gvec; % T/m/s (?. check)
GMAX  = GMAX*Gvec;  % %mT/mm

n = round((TE-TimeToEcho) * 1e-3/T);

tINV = floor((n + TimeToEcho/(T*1e3))/2);

Tzeros = [1, 2:TimeToEncode*(1e-3/T), ...
          tINV-floor(RFdur*(1e-3/T)/2):tINV+floor(RFdur*(1e-3/T)/2), n];
      
switch MMT
    case 0
        mvec = 0;
    case 1
        mvec = [0;0;];
    case 2
        mvec = [0;0;0;];
end

Nm = size(mvec, 1);
tvec = 0 + [0:n-1]*T; % in sec
% tMat for all moments
tMat = zeros( Nm, n );
for mm=1:Nm,
    tMat( mm, : ) = tvec.^(mm-1);
end
GAMMA = 42.58e3; %Hz/mT for protons
INV = ones(n,1);   INV(tINV:end) = -1;
%Q = (1/479.8714)*GAMMA*T*tMat*diag(INV);
Q = GAMMA*T*tMat*diag(INV);

normQ = sqrt(sum(Q.^2,2));
mvec=diag(1./normQ)*mvec;
Q = diag(1./normQ)*Q;

%Q = Q/normest(Q);
%Q = tMat*diag(INV);
%sabsQ=1./sum(abs(Q),2);
%Q=diag(sabsQ)*Q;

C1=tril(ones(n));
C2 = C1'*C1;
ivec = ones(1,n);
C = -ivec*C1*C1;

C=C/norm(C);

D = diag(-ones(n,1),0) + diag(ones(n-1,1),1);
D = sparse(0.5*D(1:end-1,:));

tau=1./(sum(abs(D),1) + sum(abs(Q),1))';

sigD=1./sum(abs(D),2);
% multiply by 10 for FASTER CONVERGENCE
sigQ=8./sum(abs(Q),2);

lenC1=size(C,1);
lenC2=size(C,2);
lenD1=size(D,1);
lenD2=size(D,2);
lenQ1=size(Q,1);
lenQ2=size(Q,2);

%This section initializes variables
G=zeros(lenC2,1);
xbar=G;
%zC=-1;%zeros(lenC1,1);
zD=zeros(lenD1,1);
zQ=zeros(lenQ1,1);

%step sizes.
theta=1;
p=1.95;
r2=100;r1=100;
epsilon=1e-4;
count=1;

tic

% loop until exit criterion are reached OR until we've reached 100,000
% iterations
while ((r2>epsilon) || (r1>epsilon)) && (count<=100000)
    
    % calculate x-tau*K'*z and then take the prox operation
    xbar=min(GMAX,max(-GMAX,G-tau.*((D'*zD)+(Q'*zQ)+C')));
    xbar(Tzeros) = 0;
    
    % calculate z+sigma*K(2*xbar-x)
    txmx    = 2*xbar-G;
    zDbuff  = zD + sigD.*(D*txmx);
    zQbuff  = zQ + sigQ.*(Q*txmx);
    
    % calculate the prox operators associated with z
    zDbar = zDbuff - sigD.*min(SRMAX,max(-SRMAX,zDbuff./sigD));
    zQbar = zQbuff - sigQ.*min(mvec,max(-mvec,zQbuff./sigQ));
    
    % calculate exit criterion
    if rem(count,5000)==0
        xm=G;
        zm=[zD;zQ];
    end
    
    % overrelaxation terms
    G=p*xbar+(1-p)*G;
    zD=p*zDbar+(1-p)*zD;
    zQ=p*zQbar+(1-p)*zQ;
 
    % test exit criterion
    if rem(count,5000)==0
        xtrubar=G+theta*(G-xm);
        r1=norm((xm-G)./tau)/norm(G);
        r2=norm((zm-[zD;zQ])./[sigD;sigQ] + [D*(xtrubar-G);Q*(xtrubar-G)])/norm([zD;zQ]);
    end

    % count the number of iterations
    count=count+1;
end

% check moments
M = Q*G;

if max(abs(M)) > 1e-4
    fprintf('Moments not fully nulled. Increase TE. \n');
end


end

%%

function G2 = scale_Gmax(G,b,T,tINV)
% scales the GMAX of a function that has TOO LARGE a b-value to match the
% correct b-value
% G is the input gradient waveform,
% b the desired b-value,
% T the timestep
% tINV the time of inversion

n = length(G);

% prepare calculations
GAMMA = 42.58e3; %Hz/mT for protons
C=tril(ones(n));
C2 = C'*C;
INV = ones(n,1);    INV(tINV:end) = -1;

% check initial b-value

b_val = (GAMMA*2*pi)^2*(G.*INV*T)'*(C2*(G.*INV*T))*T;

if b_val <= b
    % inital waveform was fine (or too small)
    G2 = G;
    return;
end

% scale down from 0.9, 0.8, ... etc. until the b-value is close

scales = [0.1:0.01:0.99];

bs = zeros(size(scales));

for j = 1:length(scales)
    Gtmp = G*scales(j);
    bs(j) = (GAMMA*2*pi)^2*(Gtmp.*INV*T)'*(C2*(Gtmp.*INV*T))*T;
end

bdif = abs(bs - b);

[x,ind] = min(bdif);

G2 = G*scales(ind);

end

%%
function g = trapTransform(f,G0,SR,tFact,gap,gpos)
% gradient waveform from trapezoidal reduction
% define waveform in terms of gradient duration and sign assuming 
% G = Gmax 
%
% input: f  -- row of numbers indicating the duration of each gradient lobe
%              in ms. Must correspond to an integer number of timepoints
%        G0 -- Gmax. All lobes assumed to be at Gmax
%        SR -- Slew duration (normalized to unit size) (default- 1)
%        tFact  -- Temporal resolution subsampling (default- 1)
%        gap-- Gap duration an RF pulse [units] (default 0)
%        gpos- Position of gap (list the number of f entry to put the gap
%              AFTER (default floor(num entries/2)
% 
% output: g -- fully represented gradient waveform

if nargin<2
    G0 = 0.074;
end

if nargin<3
    SR = 1;
end

if nargin<4
    tFact = 1;
end

if nargin<5
    gap = 0;
end

if nargin<6
    gpos = floor(length(f)/2);
end

if tFact == 1e-3
    tFact = 1;
    fprintf('Assuming T = 1ms, subsample of 1 used!! \n');
end

if min(abs(f)) < 1
    fprintf('ERROR - Need to allow time for slewing!!!\n');
    return;
end

%g = G0*ones( (sum(abs(f))+gap )*tFact,1);
g = G0*ones( (sum(abs(f)) + gap + 2*numel(f)*SR - (numel(f)-1) )*tFact,1);

count = 1;

for j=1:length(f)
    PLAT = abs(f(j));
    if j == gpos
        tnow = count;
        % ramp up
        g(tnow:tnow+SR-1) = g(tnow:tnow+SR-1).*(0:1/SR:1-1/SR)'*(f(j)/PLAT);
        %g(count+1:count+1+SR) = g(count+1:count+1+SR).*(0:1/SR:1)'*(f(j)/PLAT);
        tnow = tnow + SR;
        % plateau
        g(tnow:tnow+PLAT*tFact-1) = g(tnow:tnow+PLAT*tFact-1)*f(j)/PLAT;
        %g(count+2+SR:count+PLAT*tFact-SR+1) = g(count+2+SR:count+PLAT*tFact-SR+1)*f(j)/PLAT;
        tnow = tnow + PLAT*tFact;
        % ramp down
        g(tnow:tnow+SR-1) = g(tnow:tnow+SR-1).*(1-(1/SR:1/SR:1))'*(f(j)/PLAT);
       
        count = tnow + SR-1;
        
        g(count+1:count+gap*tFact) = g(count+1:count+gap*tFact)*0;
        count = count + gap*tFact;
    else
        tnow = count;
        % ramp up
        g(tnow:tnow+SR-1) = g(tnow:tnow+SR-1).*(0:1/SR:1-1/SR)'*(f(j)/PLAT);
        %g(count+1:count+1+SR) = g(count+1:count+1+SR).*(0:1/SR:1)'*(f(j)/PLAT);
        tnow = tnow + SR;
        % plateau
        g(tnow:tnow+PLAT*tFact-1) = g(tnow:tnow+PLAT*tFact-1)*f(j)/PLAT;
        %g(count+2+SR:count+PLAT*tFact-SR+1) = g(count+2+SR:count+PLAT*tFact-SR+1)*f(j)/PLAT;
        tnow = tnow + PLAT*tFact;
        % ramp down
        g(tnow:tnow+SR-1) = g(tnow:tnow+SR-1).*(1-(1/SR:1/SR:1))'*(f(j)/PLAT);
       count = tnow + SR-1;
        % ramp up
%         g(count+1:count+1+SR) = g(count+1:count+1+SR).*(0:1/SR:1)'*(f(j)/PLAT);
%         % plateau
%         g(count+1+SR:count+PLAT*tFact-SR) = g(count+1+SR:count+PLAT*tFact-SR)*f(j)/PLAT;
%         % ramp down
%         g(count+PLAT*tFact-SR:count+PLAT*tFact) = g(count+PLAT*tFact-SR:count+PLAT*tFact).*(1-(0:1/SR:1))'*(f(j)/PLAT);
%         
        %count = count + PLAT*tFact;
    end
end

end