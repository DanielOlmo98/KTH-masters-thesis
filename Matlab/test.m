%% RF signal sim
addpath('C:\Users\emine\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\MUST');
param = getparam('P4-2v');
I = imread('s12.png');

[nl,nc,~] = size(I);

L = 20e-2;
param.c = 1540; % speed of sound (m/s)
param.Nelements = 64;
% param.kerf = 0;
% param.width = 1e-4;
% param.pitch =  param.width;

lambda = param.c/param.fc;
[xi,zi] = meshgrid(linspace(0,L,nc)*nc/nl,linspace(0,L,nl));
xi = xi-L/2*nc/nl; % recenter xi

scatdens = 1; % scatterer density per lambda^2 (you may modify it)
Ns = round(scatdens*L^2*nc/nl/lambda^2); % number of scatterers

x = rand(1,Ns)*L-L/2; % scatterer locations
z = rand(1,Ns)*L;

Ig = rgb2gray(I); % convert the RGB image to gray

F = scatteredInterpolant(xi(:),zi(:),double(Ig(:))/255);
g = 0.1; % this parameter adjusts the RC values
RC = F(x,z).^(1/g); % reflection coefficients

figure;
scatter(x*1e2,z*1e2,2,abs(RC).^.15,'filled')
colormap hot
axis equal ij
set(gca,'XColor','none','box','off')
title('Scatterers')
ylabel('[cm]')

RF = cell(17,1); % this cell will contain the RF series
param.fs = 4*param.fc; % sampling frequency in Hz

%%
lines = 12;
gridsize = nl;
tilt = linspace(-pi/6,pi/6,lines); % tilt angles
IQc = zeros(gridsize,gridsize,'like',1i); % will contain the compound I/Q

opt.WaitBar = false; % no progress bar for SIMUS
param.fs = param.fc*4; % RF sampling frequency
[xI,zI] = impolgrid(gridsize,L-(1e-8),pi/3,param); % polar-type grid

h = waitbar(0,'SIMUS & DAS...');
for k = 1:lines
    dels = txdelay(param,tilt(k),pi/3); % transmit delays
    RF = simus(x,z,RC,dels,param,opt); % RF simulation
    IQ = rf2iq(RF,param); % I/Q demodulation
    IQb = das(IQ,xI,zI,dels,param); % DAS beamforming
    IQc = IQc+IQb; % compounding
    waitbar(k/length(tilt),h,...
         ['SIMUS & DAS: ' int2str(k) ' of ' int2str(lines) ' completed'])
end
close(h)

%% compound img

figure;
lcI = bmode(IQc,50); % log-compressed image
im = pcolor(xI*1e2,zI*1e2,lcI);
shading interp, axis ij square
colorbar('off')
colormap gray

set(gca,'Color','k')
set(gca,'xtick',[])
set(gca,'ytick',[])

export_fig 001test.png -grey -r130

