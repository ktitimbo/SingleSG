% SG analysis -- Geometric optics
%% Preamble
clearvars; clc; 
close all;

%%
% Choose kis to run for CQD (For this script, set to single value)
% kis = [0.10e-6,0.20e-6,0.30e-6,0.40e-6,0.50e-6,0.60e-6,0.70e-6,0.80e-6,0.90e-6,1.00e-6];
% kis = [1.10e-6,1.20e-6,1.30e-6,1.40e-6,1.50e-6,1.60e-6,1.70e-6,1.80e-6,1.90e-6,2.00e-6];
% kis = [2.10e-6,2.20e-6,2.30e-6,2.40e-6,2.50e-6,2.60e-6,2.70e-6,2.80e-6,2.90e-6,3.00e-6];
% kis = [3.10e-6,3.20e-6,3.30e-6,3.40e-6,3.50e-6,3.60e-6,3.70e-6,3.80e-6,3.90e-6,4.00e-6];
% kis = [4.10e-6,4.20e-6,4.30e-6,4.40e-6,4.50e-6,4.60e-6,4.70e-6,4.80e-6,4.90e-6,5.00e-6];
% kis = [5.10e-6,5.20e-6,5.30e-6,5.40e-6,5.50e-6,5.60e-6,5.70e-6,5.80e-6,5.90e-6,6.00e-6];
% kis = [6.10e-6,6.20e-6,6.30e-6,6.40e-6,6.50e-6,6.60e-6,6.70e-6,6.80e-6,6.90e-6,7.00e-6];
% kis = [7.10e-6,7.20e-6,7.30e-6,7.40e-6,7.50e-6,7.60e-6,7.70e-6,7.80e-6,7.90e-6,8.00e-6];

kis = 0.1e-6:0.2e-6:8.0e-6 ; 

nki = numel(kis);

% QM will assume ki=1e0

%%
% Choose # of MC runs
n_run = 3000 ;

% Currents from experiment
Icoils = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010,...
     0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100,...
     0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.700,...
     0.800, 0.900, 1.00];
nI = numel(Icoils);

% --- Make a run folder safely ---
run_stamp = "matlab_" + string(datetime('now','TimeZone','local','Format','yyyyMMdd''T''HHmmssSSS'));
outdir = fullfile(pwd, run_stamp);
if ~isfolder(outdir), mkdir(outdir); end

% Save a copy of the script
copyfile(mfilename('fullpath') + ".m", fullfile(outdir, "executedscript.m"));
fprintf('The current run is labeled as: %s\n', run_stamp);

%%
tic

% Use camera pixel size to generate histogram bins
bin = 2 ;                   % Camera binning setting of 4 was used.
px_size = 6.5e-3;           % pixel size [mm]

% --- Pixel grid setup (mm) ---
dz_bin = bin * px_size;   % bin size per pixel

% Image dimensions (after binning)
Nz = 2560 / bin;          % z-direction pixels
Nx = 2160 / bin;          % x-direction pixels

% Bin centers (z and x)
z_position = dz_bin * (-Nz/2 : Nz/2)'; 
x_position = dz_bin * (-Nx/2 : Nx/2)';

% Bin edges (midpoints between centers)
z_edges = [z_position(1) - dz_bin/2; (z_position(1:end-1) + z_position(2:end))/2; z_position(end) + dz_bin/2];
x_edges = [x_position(1) - dz_bin/2; (x_position(1:end-1) + x_position(2:end))/2; x_position(end) + dz_bin/2];

% Gaussian kernel thickness (in mm) used in the convolution (~wire thickness)
wd_sim = 0.150;

% convolve sim data with wire thickness
det_fn_sim = exp(-0.5 * (z_position / wd_sim).^2) / (sqrt(2*pi) * wd_sim);

% Tolerance for peak finding optimizer
optns = optimset('TolX',1e-10);

%%
fprintf('There are N=%d atoms, binning nz = %d, gaussian thickness %d um\n', n_run, bin, 1e3*wd_sim);

% Run simulation for each current
for iI = 1:nI
    I_now = Icoils(iI);
    fprintf('\n----- (%d/%d) Coil Current I = %.1f mA -----\n', iI, nI, 1e3*I_now);

    % Run simulation for each ki (keep it a single value for this script)
    for iki = 1:nki
        ki = kis(iki);
        fprintf('\t(%d/%d) ki = %.2e\n', iki, nki, ki);

        %%% Simulation
        % Run correlated QM and CQD for only clean peak
        [initialstate_zqm_zcqd] = SG_ode_correlatedMC(n_run, ki, Icoils(iI));
               
        % % % % % Extract final positions on detector in mm
        % % % % x_final = initialstate_zqm_zcqd(:,end-4) * 1e3;
        % % % % zcqd2   = initialstate_zqm_zcqd(:,end) * 1e3;
        % % % % zqmF1   = initialstate_zqm_zcqd(:,end-3:end-1) * 1e3;
        % % % % 
        % % % % %%% QM 
        % % % % % Generate qm histograms
        % % % % qmdata2(:,iI,iki)       = histcounts(zqmF1(:), z_edges);
        % % % % qmdata2_2d(:,:,iI,iki)  = histcounts2(repmat(x_final(:),[size(zqmF1,2),1]), zqmF1(:), x_edges, z_edges);
        % % % % 
        % % % % % Convolution to smoothen qm to find peaks w/o issues, 0.1mm kernel
        % % % % qmdata2_lpf(:,iI,iki) = conv(qmdata2(:,iI,iki),  det_fn_sim, 'same');
        % % % % 
        % % % % % Find peak pixel in qm data (clean, F=1, right lobe)
        % % % % [~,indmaxprom]          = max(qmdata2_lpf(:,iI,iki));
        % % % % locs_qm_pixel2(iI,iki)  = z_position(indmaxprom);
        % % % % 
        % % % % % Interpolate the convolved qm data and find peak (clean, F=1, right lobe)
        % % % % qminterpolant_b     = griddedInterpolant(z_position, qmdata2_lpf(:,iI,iki), "spline");
        % % % % locs_qm2(iI,iki)    = fminsearch(@(x) -qminterpolant_b(x),locs_qm_pixel2(iI,iki),optns);
        % % % % 
        % % % % % Spline fit qm and find peak (clean, F=1, right lobe)
        % % % % qmdatasplinefit2{iI,iki}    = fit(z_position,qmdata2(:,iI,iki),'smoothingspline','SmoothingParam',0.99);
        % % % % locs_qm_splinefit2(iI,iki)  = fminsearch(@(x) -qmdatasplinefit2{iI,iki}(x),locs_qm_pixel2(iI,iki),optns);
        % % % % 
        % % % % %%% CQD 
        % % % % % Generate cqd histograms
        % % % % cqddata2(:,iI,iki)      = histcounts(zcqd2(:), z_edges);
        % % % % cqddata2_2d(:,:,iI,iki) = histcounts2(repmat(x_final(:),[size(zcqd2,2),1]), zcqd2(:), x_edges, z_edges);
        % % % % 
        % % % % % Convolution to smoothen cqd to find peaks w/o issues, 0.1mm kernel
        % % % % cqddata2_lpf(:,iI,iki) = conv(cqddata2(:,iI,iki),  det_fn_sim, 'same');
        % % % % 
        % % % % % Find peak pixel in cqd data (clean, F=1, right lobe)
        % % % % [~,indmaxprom]          = max(cqddata2_lpf(:,iI,iki));
        % % % % locs_cqd_pixel2(iI,iki) = z_position(indmaxprom);
        % % % % 
        % % % % % Interpolate the convolved cqd data and find peak (clean, F=1, right lobe)
        % % % % cqdinterpolant_b    = griddedInterpolant(z_position, cqddata2_lpf(:,iI,iki), "spline");
        % % % % locs_cqd2(iI,iki)   = fminsearch(@(x) -cqdinterpolant_b(x),locs_cqd_pixel2(iI,iki),optns);
        % % % % 
        % % % % % Spline fit cqd and find peak (clean, F=1, right lobe)
        % % % % cqddatasplinefit2{iI,iki}   = fit(z_position,cqddata2(:,iI,iki),'smoothingspline','SmoothingParam',0.99);
        % % % % locs_cqd_splinefit2(iI,iki) = fminsearch(@(x) -cqddatasplinefit2{iI,iki}(x),locs_cqd_pixel2(iI,iki),optns);

        % --- Save per-run data ---
        fname = sprintf('initialstates_zqm_zcqd_ki%.2fem6_I%dmA.csv', 1e6*ki, round(1e3*I_now));
        fname = regexprep(fname, '[^\w\.-]', '_');
        writematrix(initialstate_zqm_zcqd, fullfile(outdir, fname));
        toc
    end
end
fprintf('\nTotal elapsed time: %.1f seconds\n', toc);

%%

output_CQD = [Icoils' locs_cqd2, locs_cqd_pixel2, locs_cqd_splinefit2];
output_QM  = [Icoils' locs_qm2, locs_qm_pixel2, locs_qm_splinefit2];

writematrix(output_CQD, fullfile(outdir, sprintf('results_CQD_%s.csv', run_stamp)));
writematrix(output_QM,  fullfile(outdir, sprintf('results_QM_%s.csv',  run_stamp)));

%%

% Save the results
% save(strcat(string(run_stamp), [ '/allworkspace.mat']));


%%
function [initialstate_zqm_zcqd] = SG_ode_correlatedMC(n_runs, ki, coilcurrent)
% SG_ODE_CORRELATEDMC
% Simulates correlated QM and CQD trajectories for a Stern–Gerlach setup.
% Returns particle initial states and final detector positions.


% --- Physical parameters ---
hbar        = 6.62607015e-34/2/pi;
k           = 1.380649e-23; % m^2 kg s^-2 K^-1
ps.gamma_e  = -1.76085962784e11;
ps.ge       = -2.00231930436092;
ps.uB       = 9.2740100657e-24; % J/T
ps.ue       = 9.2847646917e-24; %J/T
% --- Potassium 39 data ---
ps.gamma_n  = 1.2499416175427152e7;
ps.II       = 3/2;
nu_hfs      = 230.8598601e6;
ps.M        = 38.96370668 * 1.66053906892e-27; % kg
E_hfs       = nu_hfs * (ps.II + 1/2) * 2 * pi * hbar;
% --- Furnace ---
T     = 273 + 205; % K
alpha = sqrt(2*k*T / ps.M); % m/s
%--- Induction ---
ps.ki = ki;

% --- Geometry ---
ws   = 100e-6; % furnace slit width z
ls   = 2e-3; % furnace slit length x
wSG1 = 300e-6; % SG1 slit width z
lSG1 = 4e-3; % SG1 slit length x
ps.d1   = 224e-3; % between furnace and SG1 slit;
ps.d2   = 44e-3; % between SG1 slit and SG1 magnet
ps.d3   = 320e-3; % between SG1 exit and detector
ps.dSG  = 70e-3; % SG1 length

% Gradient vs current table from manual table
gradientvscurrent = [
    0       0;
    0.095   25.6;
    0.2     58.4;
    0.302   92.9;
    0.405   132.2;
    0.498   164.2;
    0.6     196.3;
    0.7     226;
    0.75    240;
    0.8     253.7;
    0.902   277.2;
    1.01    298.6];

% Interpolate gradient for the given current
ps.dBzdz = interp1(gradientvscurrent(:,1),gradientvscurrent(:,2),coilcurrent,'makima','extrap');

% Read the field vs current data from manual plot
fieldvscurrent = readmatrix('SG_BvsI.csv');

% Interpolate field for the given current
ps.B0 = interp1(fieldvscurrent(:,1),fieldvscurrent(:,2),coilcurrent,'makima','extrap');

% Calculate effective mu for each eigenstate using Breit-Rabi
% mu_at_B0 = zeros(8,1);  % 4*I + 2 = 8 for I=3/2
normalizedB0 = hbar * ps.B0 / (E_hfs) * (ps.gamma_e - ps.gamma_n);
% F=2 mF=2
mu_at_B0(1) = ps.ge/2 * (1 + 2 * ps.gamma_n/ps.gamma_e * ps.II);
% F=2 -1<mF<1
for mF = 1:-1:-1
    mu_at_B0(3-mF) = ps.ge * (mF*ps.gamma_n/ps.gamma_e + (1-ps.gamma_n/ps.gamma_e)/sqrt(1-4*mF*normalizedB0/(2*ps.II+1)+normalizedB0^2)*(mF/(2*ps.II+1)-normalizedB0/2));
end
% F=2 mF=-2
mu_at_B0(5) = - ps.ge/2 * (1 + 2 * ps.gamma_n/ps.gamma_e * ps.II);
% F=1 -1<mF<1
for mF = 1:-1:-1
    mu_at_B0(7-mF) = ps.ge * (mF*ps.gamma_n/ps.gamma_e - (1-ps.gamma_n/ps.gamma_e)/sqrt(1-4*mF*normalizedB0/(2*ps.II+1)+normalizedB0^2)*(mF/(2*ps.II+1)-normalizedB0/2));
end

%from here changes

% Initially all atoms alive (logical mask)
alive = true(n_runs,1);

% Random sampling of the speed from effusive beam pdf v^3*exp(-v^2/alpha^2)
v = sqrt(gammaincinv(rand(n_runs,1), 2) .* alpha^2);
% Sample initial position
z0 = (rand(n_runs,1)-0.5).*ws;
x0 = (rand(n_runs,1)-0.5).*ls;
y0 = zeros(size(z0));

% Sample initial (polar/tangential) angle = atan(vy/vtransverse)
max_angle = 1.1*atan2(sqrt((ws/2+wSG1/2)^2+(ls/2+lSG1/2)^2), ps.d1);  % Angle range based on slits
th1 = asin(sin(max_angle)*sqrt(rand(n_runs,1))); % cos(theta) effusive beam
% Sample initial azimuthal angle = atan(vz/vx)
ph1 = rand(n_runs,1)*pi*2; % Uniform azimuthal angle

% Momentum in each direction at furnace
vy0 = cos(th1) .* v;
vz0 = sin(th1) .* v .* cos(ph1);
vx0 = sin(th1) .* v .* sin(ph1);

% Calculate positions at slit 1
x_at_slit1 = x0 + vx0 .* ps.d1 ./ vy0;
z_at_slit1 = z0 + vz0 .* ps.d1 ./ vy0;
y_at_slit1 = ps.d1;

% Kill if they hit slit 1
alive = alive & (abs(x_at_slit1) <= lSG1/2) & (abs(z_at_slit1) <= wSG1/2);

% Prepare only the alive atoms for ODE solver
n_aliveafterslit1 = sum(alive);
v   = v(alive);
vy0 = vy0(alive);
vx0 = vx0(alive);
vz0 = vz0(alive);
y0  = y0(alive);
x0  = x0(alive);
z0  = z0(alive);
th1 = th1(alive);
ph1 = ph1(alive);

% Contain certain variables in a struct to easily pass to ode solver
ps.vy = vy0;

% Calculate x positions at detector
x_at_detector = x0 + vx0 .* (ps.d1+ps.d2+ps.dSG+ps.d3) ./ vy0;

% Spin states to simulate
F_mF = [6 7 8];
% disp(mu_at_B0(F_mF))
% Get mu/muB from each eigenstate for each initial state
ps.m_signs = reshape(repmat(mu_at_B0(F_mF),[n_aliveafterslit1,1]),[],1);

% Sample initial theta_e isotropically (Does not matter for QM)
ps.theta_e = 2*asin(sqrt(rand(n_aliveafterslit1,1)));

% Repeat variables for correlated run for each eigenstate
ps.theta_e = repmat(ps.theta_e,[length(F_mF),1]);
ps.vy = repmat(ps.vy,[length(F_mF),1]);

% Span of y for the solver
yspan = [0 (ps.d1+ps.d2+ps.dSG+ps.d3*1.2)];
y_eval  = [ps.d1, ps.d1 + ps.d2 + ps.dSG + ps.d3];

% Concatenate initial values for ode variables
u0 = [z0'; vz0'; zeros(size(z0'))];
u0 = repmat(u0,[1,length(F_mF)]);
u0 = u0(:);

% ODE options (reuse same opts for both QM and CQD)
opts = odeset('RelTol',1e-3,'AbsTol',1e-6);

% Run QM 
% Solve the ode for all atoms at once
sol = ode23tb(@(y,u) myODEQM(y,u,ps), yspan, u0, opts); % ode23tb ode89;

% Evaluate the z positions at important y positions (SG slit and detector)
uy = deval(sol,y_eval);
uy = reshape(uy,3,[],size(uy,2));

% z values at SG slit and detector
z_atdetector = squeeze(uy(1,:,2));

% Return final results
z_atdetector = reshape(z_atdetector,[],length(F_mF)); % Convert m to mm

% Initial states and QM results
initialstate_zqm_zcqd = [x0, z0, vx0, vy0, vz0, ps.theta_e(1:end/3), x_at_detector, z_atdetector];

% Run CQD using input ki
ps.ki = ki;
% All atoms travel up (mu/muB = +1)
ps.m_signs = abs(ps.ge/2.0) * ones([n_aliveafterslit1,1]);

% Sample theta_e and theta_n from iso and apply branching condition
theta_e_up = [];
need = n_aliveafterslit1;
while numel(theta_e_up) < need
    % sample in chunks; same distribution as original
    theta_n_tmp = 2*asin(sqrt(rand(need*3,1)));
    theta_e_tmp = 2*asin(sqrt(rand(need*3,1)));
    theta_e_up  = [theta_e_up; theta_e_tmp(theta_e_tmp < theta_n_tmp)]; %#ok<AGROW>
end
ps.theta_e = theta_e_up(1:need);

% Set initial variables for CQD solver
ps.vy = ps.vy(1:end/length(F_mF),1);
u0 = u0(1:end/length(F_mF));

% Solve the CQD ode for all atoms at once
sol = ode23tb(@(y,u) myODECQD(y,u,ps), yspan, u0, opts);

% Evaluate the z positions at important y positions
uy = deval(sol,y_eval);
uy = reshape(uy,3,[],size(uy,2));

% z values at detector
z_atdetector = squeeze(uy(1,:,2))';

% Initial states, QM and CQD results
initialstate_zqm_zcqd = [initialstate_zqm_zcqd, z_atdetector];

end


function [B0_y, dBzdz_y] = getFields(y,ps)
% Field and gradient is 0 outside the magnets
in_magnet = (y >= ps.d1+ps.d2) & (y <= ps.d1+ps.d2+ps.dSG);
B0_y      = in_magnet .* ps.B0;
dBzdz_y   = in_magnet .* ps.dBzdz;
end


function du = myODECQD(y,u,ps)
% Compute derivatives for CQD trajectories (z', vz', phi')
[B0_y, dBzdz_y] = getFields(y, ps);

% Reshape ODE variables
u = reshape(u, 3, []);

% Precompute reused terms
inv_vy = 1 ./ ps.vy';                 % precompute reciprocal
omegaB = abs(ps.gamma_e .* B0_y);     % reusable field term
y_rel  = y - ps.d1 - ps.d2;           % relative position in SG

% --- z' = vz / vy ---
dz = inv_vy .* u(2,:);

% --- vz' term (tangent model) ---
kw = ps.ki .* inv_vy .* omegaB;
x  = tan(ps.theta_e' ./ 2) .* exp(-kw .* y_rel);
cos_term = (1 - x.^2) ./ (1 + x.^2);

dvz = inv_vy .* (ps.m_signs' .* ps.uB .* dBzdz_y ./ ps.M) .* cos_term;

% --- dphi/dy ---
dphi = inv_vy .* omegaB;

% Reshape for ODE
du = [dz;dvz;dphi];
du = du(:);
end

function du = myODEQM(y,u,ps)
% Get fields at current y position
[B0_y, dBzdz_y] = getFields(y,ps);

% Reshape ODE variables for derivative calculation
u = reshape(u,3,[]);

inv_vy = 1 ./ ps.vy';                 % precompute reciprocal

% z' is vz
dz(1,:)     = inv_vy  .* u(2,:);
% vz' is az 
dvz(1,:)    = inv_vy  .* (ps.m_signs' .* ps.uB * dBzdz_y / ps.M  );
% Larmor frequency to keep track of accumulated phi
dphi(1,:)   = inv_vy  .* abs(ps.gamma_e * B0_y);

% Reshape for ODE
du = [dz;dvz;dphi];
du = du(:);
end