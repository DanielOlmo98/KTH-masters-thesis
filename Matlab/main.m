%% element delays
% addpath('C:\Users\emine\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\MUST')
param = getparam('P4-2v');
tilt = linspace(-20,20,17)/180*pi; % tilt angles in rad
txdel = cell(17,1); % this cell will contain the transmit delays

for k = 1:17
    txdel{k} = txdelay(param,tilt(k),pi/3);
end

bar(txdel{1}*1e6)
xlabel('Element number')
ylabel('Delays (\mus)')
title('TX delays for a 60{\circ}-wide -20{\circ}-tilted wave')
axis tight square

%% pressure field

xi = linspace(-4e-2,4e-2,200); % in m
zi = linspace(0,10e-2,200); % in m
[xi,zi] = meshgrid(xi,zi);

option.WaitBar = false;
P = pfield(xi,zi,txdel{1},param,option);

imagesc(xi(1,:)*1e2,zi(:,1)*1e2,20*log10(P/max(P(:))))
xlabel('x (cm)')
ylabel('z (cm)')
title('RMS pressure field for a 60{\circ}-wide -20{\circ}-tilted wave')
axis equal tight
caxis([-20 0]) % dynamic range = [-20,0] dB
c = colorbar;
c.YTickLabel{end} = '0 dB';
colormap(hot)

%% RF signal sim
param = getparam('P4-2v');
I = imread('s2.png');

[nl,nc,~] = size(I);

L = 5e-2;
param.c = 1540; % speed of sound (m/s)
lambda = param.c/param.fc;
[xi,zi] = meshgrid(linspace(0,L,nc)*nc/nl,linspace(0,L,nl));
xi = xi-L/2*nc/nl; % recenter xi

scatdens = 2; % scatterer density per lambda^2 (you may modify it)
Ns = round(scatdens*L^2*nc/nl/lambda^2); % number of scatterers

x = rand(1,Ns)*L-L/2; % scatterer locations
z = rand(1,Ns)*L;

Ig = rgb2gray(I); % convert the RGB image to gray

F = scatteredInterpolant(xi(:),zi(:),double(Ig(:))/255);
g = 0.5; % this parameter adjusts the RC values
RC = F(x,z).^(1/g); % reflection coefficients


scatter(x*1e2,z*1e2,2,abs(RC).^.15,'filled')
colormap hsv
axis equal ij
set(gca,'XColor','none','box','off')
title('Scatterers for a left ventricular 3-chamber view')
ylabel('[cm]')

RF = cell(17,1); % this cell will contain the RF series
param.fs = 4*param.fc; % sampling frequency in Hz

tilt = linspace(-pi/6,pi/6,21); % tilt angles
IQc = zeros(128,128,'like',1i); % will contain the compound I/Q

opt.WaitBar = false; % no progress bar for SIMUS
[xI,zI] = impolgrid(128,4.5e-2,pi/3,param); % polar-type grid



option.WaitBar = false; % remove the wait bar of SIMUS
h = waitbar(0,'');
for k = 1:17
    waitbar(k/17,h,['SIMUS: RF series #' int2str(k) ' of 17'])
    RF{k} = simus(x,z,RC,txdel{k},param,option);
end
close(h)



%% demodulate RF

IQ = cell(17,1);  % this cell will contain the I/Q series

for k = 1:17
    IQ{k} = rf2iq(RF{k},param.fs,param.fc);
end


%% Beamform

[xi,zi] = impolgrid([256 128],11e-2,pi/3,param);


bIQ = zeros(256,128,17);  % this array will contain the 17 I/Q images

h = waitbar(0,'');
for k = 1:17
    waitbar(k/17,h,['DAS: I/Q series #' int2str(k) ' of 17'])
    bIQ(:,:,k) = das(IQ{k},xi,zi,txdel{k},param);
end
close(h)


bIQ = tgc(bIQ); % TGC


%% show img

I = bmode(bIQ(:,:,1),50); % log-compressed image
pcolor(xi*1e2,zi*1e2,I)
shading interp, colormap gray
title('DW-based echo image with a tilt angle of -20{\circ}')

axis equal ij
set(gca,'XColor','none','box','off')
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-50 dB','0 dB'};
ylabel('[cm]')


%% compound img

cIQ = sum(bIQ,3); % this is the compound beamformed I/Q
I = bmode(cIQ,50); % log-compressed image
pcolor(xi*1e2,zi*1e2,I)
shading interp, colormap gray
title('Compound DW-based cardiac echo image')

axis equal ij
set(gca,'XColor','none','box','off')
c = colorbar;
c.YTick = [0 255];
c.YTickLabel = {'-50 dB','0 dB'};
ylabel('[cm]')
















