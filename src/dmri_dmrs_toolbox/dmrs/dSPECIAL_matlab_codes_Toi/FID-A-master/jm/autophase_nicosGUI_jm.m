function [out,phaseShift]=autophase_nicosGUI_jm(in,LB,ptsmin,ptsmax,type);

phasevect=-180:1:179; %deg
xmin_ph=ptsmin;%find(round(in.ppm,2)==round(ppmmax,2),1); %in points
xmax_ph=ptsmax;%4096;%find(round(in.ppm,2)==round(ppmmin,2),1); %in points
phasetype=type;%'sum' or 'max'

disp(in.date)

sw=round(in.spectralwidth); %Hz
dt=in.dwelltime;
np=in.n; %nb of complex points
time_vect=0:dt:(np-1)*dt; 
nrep=in.averages
    
%preprocessing
fid_ini=in.fids.';
fid_ini_lb=fid_ini.*repmat(exp(-LB*pi*time_vect),nrep,1);

ft_ini=real(fftshift(fft(fid_ini,[],2),2));    
ft_ini_lb=real(fftshift(fft(fid_ini_lb,[],2),2));
    
%autophase
for specnb=1:nrep
    nbphasesteps=length(phasevect);

    repmat_fid=repmat(fid_ini_lb(specnb,:),nbphasesteps,1);
    repmat_ph=repmat(exp(-1i.*deg2rad(phasevect')),1,np);

    prod_fid_ph=repmat_fid.*repmat_ph;

    ft_prod_fid_ph=fftshift(fft(prod_fid_ph,[],2),2);
%     size(ft_prod_fid_ph)
     
%     xmin_ph
%     xmax_ph
    
    if phasetype=='max'
        val_spec_max=max(real(ft_prod_fid_ph(:,xmin_ph:xmax_ph)).'); 
    elseif phasetype=='sum'
        val_spec_max=sum(real(ft_prod_fid_ph(:,xmin_ph:xmax_ph)).'); 
    else
        disp('pb here')
    end 
    %values vector of the highest point in the spectral region 
    %xmin_ph:xmax_ph for each value of phase
    %put sum instead of max here
    
%     figure; 
%     for k=1:10:360
%         plot(real(ft_prod_fid_ph(k,xmin_ph:xmax_ph)))
%         hold on 
%     end 
    
    ind_ph_max=find(val_spec_max==max(val_spec_max));
    %phase index that maximizes the value of the highest point in the 
    %spectral region xmin:xmax over the values of phase 
    phaseShift(specnb)=phasevect(ind_ph_max);

    fid_ini_ph(specnb,:)=fid_ini(specnb,:).*exp(-1i.*deg2rad(phaseShift(specnb)));
    ft_ini_ph(specnb,:)=ft_ini(specnb,:).*exp(-1i.*deg2rad(phaseShift(specnb)));
end 
out=in;
out.fids=fid_ini_ph.';
out.specs=ft_ini_ph.';
 