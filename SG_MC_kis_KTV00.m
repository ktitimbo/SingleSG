% SG analysis -- Geometric optics
close all;
clear all;

%%
% Choose kis to run for CQD (For this script, set to single value)
% kis = [0.10e-6,0.20e-6,0.30e-6,0.40e-6,0.50e-6,0.60e-6,0.70e-6,0.80e-6,0.90e-6,1.00e-6];
% kis = [1.10e-6,1.20e-6,1.30e-6,1.40e-6,1.50e-6,1.60e-6,1.70e-6,1.80e-6,1.90e-6,2.00e-6];
% kis = [2.10e-6,2.20e-6,2.30e-6,2.40e-6,2.50e-6,2.60e-6,2.70e-6,2.80e-6,2.90e-6,3.00e-6];
% kis = [3.10e-6,3.20e-6,3.30e-6,3.40e-6,3.50e-6,3.60e-6,3.70e-6,3.80e-6,3.90e-6,4.00e-6];
% kis = [4.10e-6,4.20e-6,4.30e-6,4.40e-6,4.50e-6,4.60e-6,4.70e-6,4.80e-6,4.90e-6,5.00e-6];
% kis = [5.10e-6,5.20e-6,5.30e-6,5.40e-6,5.50e-6,5.60e-6,5.70e-6,5.80e-6,5.90e-6,6.00e-6];
% kis = [6.10e-6,6.20e-6,6.30e-6,6.40e-6,6.50e-6,6.60e-6,6.70e-6,6.80e-6,6.90e-6,7.00e-6];
kis = [7.10e-6,7.20e-6,7.30e-6,7.40e-6,7.50e-6,7.60e-6,7.70e-6,7.80e-6,7.90e-6,8.00e-6];
% QM will assume ki=1e0

% ki indices to plot
plotiki = 1;
%%
% Choose # of MC runs
n_run = 1000 ;

% Set to 1 to save the script and the workspace
saveresults = 1;

expcurrents =  [0.00, 0.01, 0.02, 0.03, 0.05, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75];
% Currents from experiment
Icoils = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010,...
     0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100,...
     0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.600, 0.700,...
     0.800, 0.900, 1.00];

% For saving the results
if saveresults
    % Generate unique datetime based foldernamename
    strnow = string(datetime('now','TimeZone','local','Format','yyyyMMdd''T''HHmmssz'));
    % Make a new folder
    mkdir(strnow)
    % Copy the current .m script
    copyfile([mfilename('fullpath'),'.m'],strnow+"/executedscript.m");
    fprintf('The current run is labeled as: %s\n', strnow);

end

%%
tic

% Use camera pixel size to generate histogram bins
bin = 1 ;                   % Camera binning setting of 4 was used.
px_size = 6.5e-3;           % pixel size [mm]
dz_bin = bin * px_size;     % bin size
% Prepare bin z (mm)
z_position = bin*px_size*(-2560/2/bin:1:2560/2/bin)';
% Generate bin edgess for sim data similar to camera pixels
z_edges = [min(z_position)-dz_bin/2;(z_position(1:end-1)+z_position(2:end))/2;max(z_position)+dz_bin/2];
% Prepare bin x (mm)
x_position = bin*px_size*(-2160/2/bin:1:2160/2/bin)';
% Generate bin edgess for sim data similar to camera pixels
x_edges = [min(x_position)-dz_bin/2;(x_position(1:end-1)+x_position(2:end))/2;max(x_position)+dz_bin/2];

% Gaussian kernel thickness (in mm) used in the convolution (~wire thickness)
wd_sim = 0.150;
wd_exp = 0.150;

% convolve sim data with wire thickness
det_fn_sim = 1/sqrt(2*pi*(wd_sim)^2) * exp(-(z_position).^2/2/(wd_sim)^2);
det_fn_exp = 1/sqrt(2*pi*(wd_exp)^2) * exp(-(z_position).^2/2/(wd_exp)^2);

% Tolerance for peak finding optimizer
optns = optimset('TolX',1e-8);

%%
fprintf('there are N=%d atoms, binning nz = %d, gaussian thickness %d um\n', n_run, bin, 1e3*wd_sim);

% Run simulation for each current
for iI = 1:length(Icoils)

    disp("I = "+Icoils(iI)+"A")

    % Run simulation for each ki (keep it a single value for this script)
    for iki = 1:length(kis)
        ki = kis(iki);
        disp("iki = "+kis(iki)+"")

        %%% Simulation
        % Run correlated QM and CQD for only clean peak
        [initialstate_zqm_zcqd] = SG_ode_correlatedMC(n_run, ki, Icoils(iI));
        
        toc
        
        % Extract final positions on detector in mm
        x_final = initialstate_zqm_zcqd(:,end-4) * 1e3;
        zcqd2 = initialstate_zqm_zcqd(:,end) * 1e3;
        zqmF1 = initialstate_zqm_zcqd(:,end-3:end-1) * 1e3;
        
        %%% QM 
        % Generate qm histograms
        qmdata2(:,iI,iki) = histcounts(zqmF1(:), z_edges);
        qmdata2_2d(:,:,iI,iki) = histcounts2(repmat(x_final(:),[size(zqmF1,2),1]), zqmF1(:), x_edges, z_edges);

        % Convolution to smoothen qm to find peaks w/o issues, 0.1mm kernel
        qmdata2_lpf(:,iI,iki) = conv(qmdata2(:,iI,iki),  det_fn_sim, 'same');
        
        % Find peak pixel in qm data (clean, F=1, right lobe)
        [~,indmaxprom] = max(qmdata2_lpf(:,iI,iki));
        locs_qm_pixel2(iI,iki) = z_position(indmaxprom);

        % Interpolate the convolved qm data and find peak (clean, F=1, right lobe)
        qminterpolant_b = griddedInterpolant(z_position, qmdata2_lpf(:,iI,iki), "spline");
        locs_qm2(iI,iki) = fminsearch(@(x) -qminterpolant_b(x),locs_qm_pixel2(iI,iki),optns);

        % Spline fit qm and find peak (clean, F=1, right lobe)
        qmdatasplinefit2{iI,iki} = fit(z_position,qmdata2(:,iI,iki),'smoothingspline','SmoothingParam',0.99);
        locs_qm_splinefit2(iI,iki) = fminsearch(@(x) -qmdatasplinefit2{iI,iki}(x),locs_qm_pixel2(iI,iki),optns);

        %%% CQD 
        % Generate cqd histograms
        cqddata2(:,iI,iki) = histcounts(zcqd2(:), z_edges);
        cqddata2_2d(:,:,iI,iki) = histcounts2(repmat(x_final(:),[size(zcqd2,2),1]), zcqd2(:), x_edges, z_edges);

        % Convolution to smoothen cqd to find peaks w/o issues, 0.1mm kernel
        cqddata2_lpf(:,iI,iki) = conv(cqddata2(:,iI,iki),  det_fn_sim, 'same');
        
        % Find peak pixel in cqd data (clean, F=1, right lobe)
        [~,indmaxprom] = max(cqddata2_lpf(:,iI,iki));
        locs_cqd_pixel2(iI,iki) = z_position(indmaxprom);

        % Interpolate the convolved cqd data and find peak (clean, F=1, right lobe)
        cqdinterpolant_b = griddedInterpolant(z_position, cqddata2_lpf(:,iI,iki), "spline");
        locs_cqd2(iI,iki) = fminsearch(@(x) -cqdinterpolant_b(x),locs_cqd_pixel2(iI,iki),optns);

        % Spline fit cqd and find peak (clean, F=1, right lobe)
        cqddatasplinefit2{iI,iki} = fit(z_position,cqddata2(:,iI,iki),'smoothingspline','SmoothingParam',0.99);
        locs_cqd_splinefit2(iI,iki) = fminsearch(@(x) -cqddatasplinefit2{iI,iki}(x),locs_cqd_pixel2(iI,iki),optns);


        % Save initial states and final positions
        writematrix(initialstate_zqm_zcqd, string(strnow) + "/initialstates_zqm_zcqdki_1e" + string(log10(ki)) + "_I" + string(Icoils(iI)*1e3) + "mA.csv")
    end



end

%%

output_CQD = [Icoils' locs_cqd2, locs_cqd_pixel2, locs_cqd_splinefit2]
writematrix(output_CQD,string(strnow)+'/results_CQD_'+strnow+'.csv')

output_QM = [Icoils' locs_qm2, locs_qm_pixel2, locs_qm_splinefit2]
writematrix(output_QM,string(strnow)+'/results_QM_'+strnow+'.csv')

%%


% % Peak positions from Kelvin
% exp_peaks_KT = [0.00509172, 0.00124986;
%     0.0037126, 0.00900368;
%     -0.0081192, 0.0227256;
%     -0.0369902, 0.0629495;
%     -0.0848596, 0.11486;
%     -0.378723, 0.390562;
%     -0.530597, 0.510494;
%     -0.656589, 0.631897;
%     -0.860917, 0.812013;
%     -1.1428, 1.12686;
%     -1.58553, 1.59759];

% % Plot 2D data
% figure;
% tiledlayout(2,4,"TileSpacing","compact")
% for iI = [1 4 7 11]
%     nexttile;
%     imagesc(x_position,z_position,qmdata2_2d(:,:,iI,end))
%     xlabel('x (mm)');
%     ylabel('z (mm)');
%     title("F=1, I="+Icoils(iI)+"A")
% end
% for iI = [1 4 7 11]
%     nexttile;
%     imagesc(x_position,z_position,cqddata2_2d(:,:,iI,end))
%     xlabel('x (mm)');
%     ylabel('z (mm)');
%     title("CQD ki=1e" + string(log10(kis(plotiki))) + ", I="+Icoils(iI)+"A")
% end

% % Plot 1D data
% figure;
% tiledlayout(3,4,"TileSpacing","compact")
% for iI = [1 4 7 11]
%     nexttile;
%     plot(z_position,qmdata2(:,iI));
%     ylabel('Simulation');
%     xlabel('z (mm)');
%     title("QM F=1, I="+Icoils(iI)+"A")
% end
% for iI = [1 4 7 11]
%     nexttile;
%     plot(z_position,cqddata2(:,iI,plotiki));
%     ylabel('Simulation');
%     xlabel('z (mm)');
%     title("CQD ki=1e" + string(log10(kis(plotiki))) + ", I="+Icoils(iI)+"A")
% end

% figure;
% nexttile;
% plot(Icoils, locs_qm_pixel2(:,plotiki));
% hold on;
% plot(Icoils, locs_qm_pixel2(:,plotiki));
% plot(Icoils, locs_qm2(:,plotiki));
% plot(Icoils, locs_qm_splinefit2(:,plotiki));
% plot(Icoils, locs_cqd_pixel2(:,plotiki));
% plot(Icoils, locs_cqd2(:,plotiki));
% plot(Icoils, locs_cqd_splinefit2(:,plotiki));
% grid on;
% legend(["Sim pixel QM","Sim sub-pixel QM","Sim spline QM","Sim pixel ki=1e"+string(log10(kis(plotiki))),"Sim sub-pixel ki=1e"+string(log10(kis(plotiki))),"Sim spline ki=1e"+string(log10(kis(plotiki)))])
% ylabel('Right peak location (mm)')
% xlabel('Current (A)')


% figure;
% nexttile;
% hold on;
% loglog(expcurrents, exp_peaks_KT(:,2),'-x');
% loglog(Icoils, locs_qm2(:,1),'-+');
% loglog(Icoils, locs_cqd2(:,plotiki),'-sq');
% grid on;
% legend(["Exp Kelvin","Sim sub-pixel QM","Sim sub-pixel ki=1e"+string(log10(kis(plotiki)))])
% ylabel('Right peak location (mm)')
% xlabel('Current (A)')
% title('Convolved with Gaussian and interpolated')

% figure;
% nexttile;
% loglog(expcurrents, exp_peaks_KT(:,2),'-x');
% hold on;
% loglog(Icoils, locs_qm_splinefit2(:,plotiki),'-+');
% loglog(Icoils, locs_cqd_splinefit2(:,plotiki),'-sq');
% grid on;
% legend(["Exp Kelvin","Sim spline QM","Sim spline ki=1e"+string(log10(kis(plotiki)))])
% ylabel('Right peak location (mm)')
% xlabel('Current (A)')
% title("Spline fitting")


% Save the results
save(strcat(string(strnow), [ '/allworkspace.mat']));

FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
    iFig
    FigHandle = FigList(iFig);
    FigName   = "Fig"+get(FigHandle, 'Number');
    set(0, 'CurrentFigure', FigHandle);
    savefig(FigHandle, strcat(string(strnow), '/', FigName, '.fig'));
    saveas(FigHandle, strcat(string(strnow), '/', FigName, '.png'));
end


%%
function [initialstate_zqm_zcqd] = SG_ode_correlatedMC(n_runs, ki, coilcurrent)

%%% Physical parameters
ps.gamma_e = -1.76085962784e11;
ps.gamma_n = 1.2499416175427152e7;
ps.II = 3/2;
ps.ge = -2.00231930436118;
nu_hfs = 230.8598601e6;
hbar = 6.62607015e-34/2/pi;
k = 1.38065e-23; % m^2 kg s^-2 K^-1
T = 273 + 205; % K
ps.M = 38.96370668 * 1.66053906892e-27; % kg
ps.uB = 9.2740100657e-24;  % J / Tesla
ps.ue = 9.284764691e-24; %J/T

E_hfs = nu_hfs * (ps.II + 1/2) * 2 * pi * hbar;
alpha = sqrt(2*k*T / ps.M); % m/s
ps.ki = ki;

% Slit thicknesses
ws   = 100e-6; % furnace slit width z
ls   = 2e-3; % furnace slit length x
wSG1 = 300e-6; % SG1 slit width z
lSG1 = 4e-3; % SG1 slit length x
wSG2 = 0.3e-3; % SG2 slit
lSG2 = 4e-3; % SG1 slit length x

% y locations
ps.d1 = 224e-3; % between furnace and SG1 slit;
ps.d2 = 44e-3; % between SG1 slit and SG1 magnet
d3 = 320e-3; % between SG1 exit and detector
ps.d_SG = 70e-3; % SG1 length

% Gradient vs current table from manual table
gradientvscurrent = [0 0;
    0.095 25.6;
    0.2 58.4;
    0.302 92.9;
    0.405 132.2;
    0.498 164.2;
    0.6 196.3;
    0.7 226;
    0.75 240;
    0.8 253.7;
    0.902 277.2;
    1.01 298.6];

% Interpolate gradient for the given current
ps.dBzdz = interp1(gradientvscurrent(:,1),gradientvscurrent(:,2),coilcurrent,'linear','extrap');

% Read the field vs current data from manual plot
fieldvscurrent = readmatrix('SG_BvsI.csv');

% Interpolate field for the given current
ps.B0 = interp1(fieldvscurrent(:,1),fieldvscurrent(:,2),coilcurrent,'makima','extrap');

% Calculate effective mu for each eigenstate using Breit-Rabi
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

% Initially all atoms alive
alive = ones(n_runs,1);

% Random sampling of the speed from effusive beam pdf v^3*exp(-v^2/alpha^2)
v = sqrt(gammaincinv(rand(n_runs,1), 2) .* alpha^2);

% Sample initial position z
z0 = (rand(n_runs,1)-0.5).*ws;
% Sample initial position x
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
alive = alive .* (abs(x_at_slit1)<=lSG1/2&abs(z_at_slit1)<=wSG1/2);

% Prepare only the alive atoms for ode solver
n_aliveafterslit1 = sum(alive);
disp(n_aliveafterslit1)
v = v(alive==1,1);
vy0 = vy0(alive==1,1);
vx0 = vx0(alive==1,1);
vz0 = vz0(alive==1,1);
y0 = y0(alive==1,1);
x0 = x0(alive==1,1);
z0 = z0(alive==1,1);
th1 = th1(alive==1,1);
ph1 = ph1(alive==1,1);

% Contain certain variables in a struct to easily pass to ode solver
ps.vy = vy0;

% Calculate x positions at detector
x_at_detector = x0 + vx0 .* (ps.d1+ps.d2+ps.d_SG+d3) ./ vy0;

% Spin states to simulate
F_mF = [6 7 8];
% Get mu/muB from each eigenstate for each initial state
ps.m_signs = reshape(repmat(mu_at_B0(F_mF),[n_aliveafterslit1,1]),[],1);

% Sample initial theta_e isotropically (Does not matter for QM)
ps.theta_e = 2*asin(sqrt(rand(n_aliveafterslit1,1)));

% Repeat variables for correlated run for each eigenstate
ps.theta_e = repmat(ps.theta_e,[length(F_mF),1]);
ps.vy = repmat(ps.vy,[length(F_mF),1]);

% Span of y for the solver
yspan = [0 (ps.d1+ps.d2+ps.d_SG+d3*1.2)];

% Concatenate initial values for ode variables
u0 = [z0'; vz0'; zeros(size(z0'))];
u0 = repmat(u0,[1,length(F_mF)]);
u0 = u0(:);

% Run QM using high ki
ps.ki = 1e0;

% Solve the ode for all atoms at once
sol = ode89(@(y,u) myODE(y,u,ps), yspan, u0);

% Evaluate the z positions at important y positions (SG slit and detector)
uy = deval(sol,[ps.d1 ps.d1+ps.d2+ps.d_SG+d3]);
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
ps.m_signs = ones([n_aliveafterslit1,1]);

% Sample theta_e and theta_n from iso and apply branching condition
theta_e_up = [];
theta_e_down = [];
% Keep sampling and filtering until we have sufficient atoms
while (length(theta_e_up)<n_aliveafterslit1) && (length(theta_e_down)<n_aliveafterslit1)
    % Iso sample
    theta_n_tmp = 2*asin(sqrt(rand(n_aliveafterslit1*3,1)));
    theta_e_tmp = 2*asin(sqrt(rand(n_aliveafterslit1*3,1)));
    % Split using branching condition
    theta_e_up = [theta_e_up;theta_e_tmp(theta_e_tmp < theta_n_tmp)];
    theta_e_down = [theta_e_down;theta_e_tmp(theta_e_tmp > theta_n_tmp)];
end
% Take only n_aliveafterslit1 samples
theta_e_up = theta_e_up(1:n_aliveafterslit1,1);
theta_e_down = theta_e_down(1:n_aliveafterslit1,1);
% Use only the atoms going up
ps.theta_e = theta_e_up;

% Set initial variables for CQD solver
ps.vy = ps.vy(1:end/length(F_mF),1);
u0 = u0(1:end/length(F_mF));

% Solve the CQD ode for all atoms at once
sol = ode89(@(y,u) myODE(y,u,ps), yspan, u0);

% Evaluate the z positions at important y positions
uy = deval(sol,[ps.d1 ps.d1+ps.d2+ps.d_SG+d3]);
uy = reshape(uy,3,[],size(uy,2));

% z values at detector
z_atdetector = squeeze(uy(1,:,2))';

% Initial states, QM and CQD results
initialstate_zqm_zcqd = [initialstate_zqm_zcqd, z_atdetector];

end


function [B0_y, dBzdz_y] = getFields(y,ps)
% Field and gradient is 0 outside the magnets
B0_y = (y>ps.d1+ps.d2&&y<ps.d1+ps.d2+ps.d_SG) .* ps.B0;
dBzdz_y = (y>ps.d1+ps.d2&&y<ps.d1+ps.d2+ps.d_SG) .* ps.dBzdz;
end

function du = myODE(y,u,ps)
% Get fields at current y position
[B0_y, dBzdz_y] = getFields(y,ps);

% Reshape ODE variables for derivative calculation
u = reshape(u,3,[]);

% z' is vz
dz(1,:) = (1./ps.vy') .* u(2,:);
% vz' is az calculated using the tangent equation
dvz(1,:) = (1./ps.vy') .* (- ps.m_signs' .* ps.ue * dBzdz_y / ps.M .* cos( 2*atan( tan(ps.theta_e'/2).*exp(ps.ki*u(3,:)) ) ) );
% Larmor frequency to keep track of accumulated phi
dphi(1,:) = (1./ps.vy') .* abs(ps.gamma_e * B0_y);

% Reshape for ODE
du = [dz;dvz;dphi];
du = du(:);
end


%%

Icoils = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,...
     0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100,...
     0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 1.000];

% Read the field vs current data from manual plot
fieldvscurrent = readmatrix('./SG_BvsI.csv');

% Interpolate field for the given current
B0 = interp1(fieldvscurrent(:,1),fieldvscurrent(:,2),Icoils,'makima','extrap');


% Gradient vs current table from manual table
gradientvscurrent = [0 0;
    0.095 25.6;
    0.2 58.4;
    0.302 92.9;
    0.405 132.2;
    0.498 164.2;
    0.6 196.3;
    0.7 226;
    0.75 240;
    0.8 253.7;
    0.902 277.2;
    1.01 298.6];

% Interpolate gradient for the given current
dBzdz = interp1(gradientvscurrent(:,1),gradientvscurrent(:,2),Icoils,'makima','extrap');
