param = getparam('P4-2v');
param.c = 1540; % speed of sound (m/s)

lambda = param.c/param.fc;

I = imread('s2.png');

[nl,nc,~] = size(I);

L = 5e-2;
[xi,zi] = meshgrid(linspace(0,L,nc)*nc/nl,linspace(0,L,nl));
xi = xi-L/2*nc/nl; % recenter xi

scatdens = 2; % scatterer density per lambda^2 (you may modify it)
Ns = round(scatdens*L^2*nc/nl/lambda^2); % number of scatterers

xs = rand(1,Ns)*L-L/2; % scatterer locations
zs = rand(1,Ns)*L;

Ig = rgb2gray(I); % convert the RGB image to gray

F = scatteredInterpolant(xi(:),zi(:),double(Ig(:))/255);
g = 0.5; % this parameter adjusts the RC values
RC = F(xs,zs).^(1/g); % reflection coefficients


%%
tilt = linspace(-pi/6,pi/6,21); % tilt angles
IQc = zeros(128,128,'like',1i); % will contain the compound I/Q

opt.WaitBar = false; % no progress bar for SIMUS
param.fs = param.fc*4; % RF sampling frequency
[xI,zI] = impolgrid(128,4.5e-2,pi/3,param); % polar-type grid

h = waitbar(0,'SIMUS & DAS...');
for k = 1:21
    dels = txdelay(param,tilt(k),pi/3); % transmit delays
    RF = simus(xs,zs,RC,dels,param,opt); % RF simulation
    IQ = rf2iq(RF,param); % I/Q demodulation
    IQb = das(IQ,xI,zI,dels,param); % DAS beamforming
    IQc = IQc+IQb; % compounding
    waitbar(k/length(tilt),h,...
        ['SIMUS & DAS: ' int2str(k) ' of 21 completed'])
end
close(h)

%% show
figure;
lcI = bmode(IQc,50); % log-compressed image
pcolor(xI*1e2,zI*1e2,lcI)
shading interp, axis equal ij tight
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-50 dB','0 dB'};
colormap gray
axis equal tight
ylabel('[cm]')
set(gca,'XColor','none','box','off')
%colormap hot
title('ultrasound compound image')
