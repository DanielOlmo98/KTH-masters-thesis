% Load entire mat file contents into a structure.
% The structure has a member "I" that is a double 512x512 array.
load('LVscat.mat');

figure;
scatter(x*1e2,z*1e2,2,abs(BSC).^.15,'filled')
colormap hot
axis equal ij
set(gca,'XColor','none','box','off')
title('Scatterers for a left ventricular 3-chamber view')
ylabel('[cm]')