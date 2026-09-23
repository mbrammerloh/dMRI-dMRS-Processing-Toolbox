%nice plot Bruker 
clear;close all;
%% open FID
%path of 1D fid file in the matlab current directory or same folder
expnb=25;
type="ser"; 
NR=192;
grpdly=77;
cutspec=2048;
dt=0.000140; %dwell time
LB=2;
swidth=7142; 
ph0=0;

fullpath="G:\38-knowledge-transfer\14T\MRS\2-Processing\Example_data\1-STEAM\raw\" + num2str(expnb) + "\" + type;

fileid=fopen(fullpath,'r','ieee-le'); %read binary format
if fileid == -1
    disp('Cannot open file');
    return
end

buffer=fread(fileid,'int32'); 
fclose(fileid);
nbptsfid=length(buffer)/(2*NR);
fid=reshape(buffer,nbptsfid*2,NR)';
fid_c=fid(:,1:2:end)+1i*fid(:,2:2:end); 

%grpdly
fid_c_shift=[fid_c(:,grpdly:end),fid_c(:,1:grpdly-1)];
 
figure; 
for k=1:192
    plot(real(fftshift(fft(fid_c_shift(k,:),[],2),2)))
   hold on 
end 

fid_c_sum=sum(fid_c_shift);

%if neeeded, phase0 correction
fid_c_sum_shift_ph=fid_c_sum.*exp(i*ph0);

%cut and zero fill
fid_c_sum_shift_ph_zerofill=[fid_c_sum_shift_ph(1:cutspec),zeros(1,nbptsfid-cutspec)];

%apodize
tt=[0:dt:(nbptsfid-1)*dt];
fid_c_sum_shift_ph_zerofill_apo=fid_c_sum_shift_ph_zerofill.*exp(-tt.*LB);

% %shift
% xx=[0:1:4096-1];
% fid_c_sum_shift_ph_zerofill_apo=fid_c_sum_shift_ph_zerofill_apo.*exp(i.*-0.005.*xx);

%ft
ft_processed=fftshift(fft(fid_c_sum_shift_ph_zerofill_apo,[],2),2);

% %1st order phase
% xxx=[-2047:1:2048];
% ft_processed=ft_processed.*exp(i.*-0.005.*xxx);


fmax=swidth/2;
f=[fmax:-2*fmax/(nbptsfid-1):-fmax];
ppmscale=f/599.41+4.7;


%plot
figure()
hax=axes;
plot(ppmscale,flip(real(ft_processed)),'color','black','LineWidth',1) 
xlim([-2,5.3])
xlabel('ppm')
set(gca, 'XDir','reverse')

set(hax,'box','off','YTick', [])
hax.YAxis.Visible = 'off'; % remove y-axis

% set(gca,'TickDir','out')
