function [] = b_Preprocessing_FidA_9p4T_EM_MB(folder_results , expnb, phi, job0)
%% b_Preprocessing_FidA_9p4T_EM
%EM no change from Toi except path version 10.11.2025
% RO Rita changed absolute paths to relative paths and used fullfile
% MB Malte replaced pwd references in Eloises script with fullfile
% references like in Rita's adaptation. 

% instead of hardcoded slashes
% clear; clc; %close all; 
% 
% folder_results   = "Z:\DATA\CSI\9.4T-FIDCSI-Data\20250424_084847_Toi_dMRSI_20250424_RatFem_dMRI_dSPECIAL_dMRSI_1_11_cc";
% 
% expnb       = 35; % scan number
%disp(['execution b_Processing_FidA_9p4T_EM (Toi version 10.11.2025) on data: ',folder_results ])

% MB: I adapted Eloises new code (EM2) to fit into this file. Original
% comments were: 
% EM2 : new argin : option for analyzing the rawdatajob0 file
% This version is compatible with a_Create_study_Bruker_9p4T_EM2
% in :
% folder_results : path to the data
% expnb : id of scan (Bruker)
% phi : zero-order phase for the last figure (after all corrections and the sum)
% job0 : 1: analysis of the job0 file / 0: analysis of the ser file / 2: analysis of the two files job0 and ser
% if job0 = 2  : figure plot only jb0 but both are analysed


% MB: We want this to fail if we forget to specify job0, so there is no
% silent change in this script, but the default case is controlled via
% python and visible.
% EM2 changed (16/02/2026)
% if nargin<3
%    %old version
%    job0 = 0;
% end
% EM2 (end)

disp(['execution b_Processing_FidA_9p4T_EM2 (Toi version 10.11.2025 updated 16.02.2026 by EM) on data: ',folder_results ,'\',num2str(expnb) ])

tic
%% step 1 - do preprocessing with FIDA

% EM2 (16/02/2026)
% filelist = dir(fullfile(folder_results, "/raw/*ser.mat"));
if job0 == 0
    filelist = dir(fullfile(folder_results, "raw","*ser.mat"));
elseif job0 == 1
    filelist = dir(fullfile(folder_results, "raw", "*rawdatajob0.mat"));
elseif job0 == 2
    filelist2 = dir(fullfile(folder_results, "raw", "*ser.mat"));
    filelist = dir(fullfile(folder_results, "raw", "*rawdatajob0.mat"));
end
% filelist = dir(fullfile(folder_results, "raw","*ser.mat"));
% EM2 (end) (16/02/2026)

idx = find(contains({filelist.name}, ['_',num2str(expnb),'_']));   % look for number 43 in name
filelist = filelist(idx,:); 

% EM2 (16/02/2026)
if job0 == 2
    idx2 = find(contains({filelist2.name}, ['_',num2str(expnb),'_']));   % look for number 43 in name  
    filelist2 = filelist2(idx2,:);
end
% EM2 (end) (16/02/2026)

% add FID-A master to Matlab path
% addpath(genpath('Y:\Toi\dMRS_SPECIAL\dSPECIAL_scripts\dSPECIAL_Manual_Codes\FID-A-master'))

LBall               = 10;
doprocessing        = true;
dosavesum           = true;
dosaveprocessing    = true; 

spectralreginf      = 0.5; %ppm
spectralregsup      = 4.3; %ppm

if doprocessing
    screenSz = get(0,'ScreenSize'); figure('Position',[0,screenSz(4)/3-150,screenSz(3)-screenSz(3)/5,screenSz(4)/3],'Color','w');
    til = tiledlayout(length(filelist),4,"Padding","tight");

    if ~exist(fullfile(folder_results, 'processed'), 'dir') && dosaveprocessing
           mkdir(fullfile(folder_results, 'processed'))
    end

for file=1:length(filelist)
    disp(filelist(file).name)
    % f1=figure;

    %% 1-Convert Bruker study to FID A structure
    % EM2 (16/02/2026)
    % out0a=convertNicoStudyFidAformat_isison(char(fullfile(folder_results,'raw')), filelist(file).name);
    % out0b=convertNicoStudyFidAformat_isisoff(char(fullfile(folder_results,'raw')), filelist(file).name);
    if job0 == 0
        out0a=convertNicoStudyFidAformat_isison(char(fullfile(folder_results,'raw')), filelist(file).name);
        out0b=convertNicoStudyFidAformat_isisoff(char(fullfile(folder_results,'raw')), filelist(file).name);
    elseif job0 == 1
        out0a=convertNicoStudyFidAformat_isison_job0(char(fullfile(folder_results,'raw')), filelist(file).name);
        out0b=convertNicoStudyFidAformat_isisoff_job0(char(fullfile(folder_results,'raw')), filelist(file).name);
    elseif job0 == 2
        out0a=convertNicoStudyFidAformat_isison_job0(char(fullfile(folder_results,'raw')), filelist(file).name);
        out0b=convertNicoStudyFidAformat_isisoff_job0(char(fullfile(folder_results,'raw')), filelist(file).name);
        out2_0a=convertNicoStudyFidAformat_isison(char(fullfile(folder_results,'raw')), filelist2(file).name);
        out2_0b=convertNicoStudyFidAformat_isisoff(char(fullfile(folder_results,'raw')), filelist2(file).name);
    end
    % EM2 (end) (16/02/2026)

    % out0a=convertNicoStudyFidAformat_isison(char(fullfile(folder_results,'raw')), filelist(file).name);
    % out0b=convertNicoStudyFidAformat_isisoff(char(fullfile(folder_results,'raw')), filelist(file).name);
    %apply LB
    dw=out0a.dwelltime;
    tt=[0:dw:dw*(out0a.n-1)];
    out0alb=out0a;
    fids0alb=out0alb.fids.*repmat(exp(-tt*pi*LBall).',1,out0a.averages);
    out0alb.fids=fids0alb;
    out0alb.specs=fftshift(fft(out0alb.fids.',[],2),2).';
    out0blb=out0b;
    fids0blb=out0blb.fids.*repmat(exp(-tt*pi*LBall).',1,out0b.averages);
    out0blb.fids=fids0blb;
    out0blb.specs=fftshift(fft(out0blb.fids.',[],2),2).';
    

    % figure(f1);
    % subplot(2,2,1)
    nexttile(til);
    for k=1:out0a.averages
        plot(real(out0a.specs(:,k)))
        hold on; xlim tight;  ylim  tight;
        % xlim([2900,3050])
    end
    for k=1:out0b.averages
        plot(real(out0b.specs(:,k)))
        hold on; xlim tight;  ylim  tight;
        % xlim([2900,3050])
    end 
    title(['E',num2str(expnb),' - raw'])
    
    % figure(f1);
    % subplot(2,2,2)
    nexttile(til);
    for k=1:out0alb.averages
        plot(real(out0alb.specs(:,k)))
        hold on; xlim tight;  ylim  tight;
        % xlim([2900,3050])
    end 
    for k=1:out0blb.averages
        plot(real(out0blb.specs(:,k)))
        hold on; xlim tight;  ylim  tight;
        % xlim([2900,3050])
    end 
    title(['E',num2str(expnb),' - Linebroadening ' num2str(LBall) 'Hz'])

    % EM2 (16/02/2026)
    if job0 == 2
        %apply LB
        dw=out2_0a.dwelltime;
        tt=[0:dw:dw*(out2_0a.n-1)];
        out2_0alb=out2_0a;
        fids2_0alb=out2_0alb.fids.*repmat(exp(-tt*pi*LBall).',1,out2_0a.averages);
        out2_0alb.fids=fids2_0alb;
        out2_0alb.specs=fftshift(fft(out2_0alb.fids.',[],2),2).';
        out2_0blb=out2_0b;
        fids2_0blb=out2_0blb.fids.*repmat(exp(-tt*pi*LBall).',1,out2_0b.averages);
        out2_0blb.fids=fids2_0blb;
        out2_0blb.specs=fftshift(fft(out2_0blb.fids.',[],2),2).';
    end
    % EM2 (end) (16/02/2026)



    %% 2-align av.
    [out1alb,fsa,phsa]=op_alignAverages_fd_jm_conj(out0alb,4.7+spectralreginf,4.7+spectralregsup,0.5,'y');%5.2,9,0.5,'y');
    [out1blb,fsb,phsb]=op_alignAverages_fd_jm_conj(out0blb,4.7+spectralreginf,4.7+spectralregsup,0.5,'y');%5.2,9,0.5,'y');
    % fs        = Vector of frequency shifts (in Hz) used for alignment.
    % phs       = Vector of phase shifts (in degrees) used for alignment.

    % figure(f1);
    % subplot(2,2,3)
    nexttile(til);
    for k=1:out1alb.averages
        plot(real(out1alb.specs(:,k)))
        hold on; xlim tight; ylim  tight;
        % xlim([2900,3050])
    end
    for k=1:out1blb.averages
        plot(real(out1blb.specs(:,k)))
        hold on; xlim tight; ylim  tight;
        % xlim([2900,3050])
    end
    title(['E',num2str(expnb),' - Spectral alignment (JNear averaged)'])
    
    % EM2 (16/02/2026)
    if job0 == 2
        [out2_1alb,fsa2,phsa2]=op_alignAverages_fd_jm_conj(out2_0alb,4.7+spectralreginf,4.7+spectralregsup,0.5,'y');%5.2,9,0.5,'y');
        [out2_1blb,fsb2,phsb2]=op_alignAverages_fd_jm_conj(out2_0blb,4.7+spectralreginf,4.7+spectralregsup,0.5,'y');%5.2,9,0.5,'y');
    end
    % EM2 (end) (16/02/2026)

    %remove LB
    out1a=out1alb;
    fids1a=out1alb.fids.*repmat(exp(tt*pi*LBall).',1,out1alb.averages);
    out1a.fids=fids1a;
    out1a.specs=fftshift(fft(out1a.fids.',[],2),2).';
    out1b=out1blb;
    fids1b=out1blb.fids.*repmat(exp(tt*pi*LBall).',1,out1blb.averages);
    out1b.fids=fids1b;
    out1b.specs=fftshift(fft(out1b.fids.',[],2),2).';

    % EM2 (16/02/2026)
    if job0 == 2
        %remove LB
        out2_1a=out2_1alb;
        fids2_1a=out2_1alb.fids.*repmat(exp(tt*pi*LBall).',1,out2_1alb.averages);
        out2_1a.fids=fids2_1a;
        out2_1a.specs=fftshift(fft(out2_1a.fids.',[],2),2).';
        out2_1b=out2_1blb;
        fids2_1b=out2_1blb.fids.*repmat(exp(tt*pi*LBall).',1,out2_1blb.averages);
        out2_1b.fids=fids2_1b;
        out2_1b.specs=fftshift(fft(out2_1b.fids.',[],2),2).';
    end
    % EM2 (end) (16/02/2026)

    %% 3-outlier removal 
    [out2a,metrica,badAveragesa]=op_rmbadaverages_jm(out1a,1.5,'f'); %performs 10Hz LB inside
    [out2b,metricb,badAveragesb]=op_rmbadaverages_jm(out1b,1.5,'f'); %performs 10Hz LB inside
    
    % EM2 (16/02/2026)
    if job0 == 2
        [out2_2a,metric2_a,badAverages2_a]=op_rmbadaverages_jm(out2_1a,1.5,'f'); %performs 10Hz LB inside
        [out2_2b,metric2_b,badAverages2_b]=op_rmbadaverages_jm(out2_1b,1.5,'f'); %performs 10Hz LB inside
    end
    % EM2 (end) (16/02/2026)

    %apply LB
    out2alb=out2a;
    fids2alb=out2a.fids.*repmat(exp(-tt*pi*LBall).',1,out2a.averages);
%     out2alb.fids=conj(fids2alb);
    out2alb.fids=fids2alb;
    out2alb.specs=fftshift(fft(out2alb.fids.',[],2),2).';
    out2blb=out2b;
    fids2blb=out2b.fids.*repmat(exp(-tt*pi*LBall).',1,out2b.averages);
    out2blb.fids=fids2blb;
    out2blb.specs=fftshift(fft(out2blb.fids.',[],2),2).';

    % EM2 (16/02/2026)
    if job0 == 2
        %apply LB
        out2_2alb=out2_2a;
        fids2_2alb=out2_2a.fids.*repmat(exp(-tt*pi*LBall).',1,out2_2a.averages);
        %     out2alb.fids=conj(fids2alb);
        out2_2alb.fids=fids2_2alb;
        out2_2alb.specs=fftshift(fft(out2_2alb.fids.',[],2),2).';
        out2_2blb=out2_2b;
        fids2_2blb=out2_2b.fids.*repmat(exp(-tt*pi*LBall).',1,out2_2b.averages);
        out2_2blb.fids=fids2_2blb;
        out2_2blb.specs=fftshift(fft(out2_2blb.fids.',[],2),2).';
    end
    % EM2 (end) (16/02/2026)
    
    % figure(f1);
    % subplot(2,2,4)
    nexttile(til);
    for k=1:out2alb.averages
        plot(real(out2alb.specs(:,k)))
        hold on; xlim tight;  ylim  tight;
        % xlim([2900,3050])
    end
    for k=1:out2blb.averages
        plot(real(out2blb.specs(:,k)))
        hold on; xlim tight;  ylim  tight;
        % xlim([2900,3050])
    end
    title(['E',num2str(expnb),' - Outlier removal']); ylim  tight;
    a=legend(gca,num2str([badAveragesa;badAveragesb]),'Box','off','Location','southwest'); title(a,'Bad averages');
    
    % Save the plot
    plotname = ['Bruker_' num2str(expnb) '.png']; 
    saveas(gcf, fullfile(folder_results, 'raw', plotname));  % Save the figure as PNG

    %combine on/off
    clear fidtot;
    fida=out1a.fids.'; 
    fidb=out1b.fids.'; 
    fidtot(1:2:size(fida,1)*2,:)=fida; 
    fidtot(2:2:size(fida,1)*2,:)=fidb;
    
    ind=1;
    clear fid2sum;
    for k=1:size(fidtot,1)/2
        fid2sum(ind,:)=sum(fidtot((k-1)*2+1:k*2,:)); 
        ind=ind+1;
    end 
    
    ind=1;
    clear fidmocor;
    for k=1:size(fidtot,1)/2 %from 1 to 80 pairs
        if ismember(k,badAveragesa) % if in pair k, the odd is an outlier (= present in the 1st outlier list) - out
        else % if in pair k, the odd is an not an outlier 
            if ismember(k,badAveragesb) % if in pair k, the even is an outlier (= present in the 2nd outlier list) - out
            else % if in pair k, neither the odd nor the even are outliers
                fidmocor(ind,:) = fid2sum(k,:);
                ind=ind+1; 
            end 
        end 
    end 

    fidmocor=conj(fidmocor);

    % EM2 (16/02/2026)
    if job0 == 2
        %combine on/off
        clear fidtot;
        fid2_a=out2_1a.fids.';
        fid2_b=out2_1b.fids.';
        fid2_tot(1:2:size(fid2_a,1)*2,:)=fid2_a;
        fid2_tot(2:2:size(fid2_a,1)*2,:)=fid2_b;

        ind=1;
        clear fid2_2sum;
        for k=1:size(fid2_tot,1)/2
            fid2_2sum(ind,:)=sum(fid2_tot((k-1)*2+1:k*2,:));
            ind=ind+1;
        end

        ind=1;
        clear fid2_mocor;
        for k=1:size(fid2_tot,1)/2 %from 1 to 80 pairs
            if ismember(k,badAverages2_a) % if in pair k, the odd is an outlier (= present in the 1st outlier list) - out
            else % if in pair k, the odd is an not an outlier
                if ismember(k,badAverages2_b) % if in pair k, the even is an outlier (= present in the 2nd outlier list) - out
                else % if in pair k, neither the odd nor the even are outliers
                    fid2_mocor(ind,:) = fid2_2sum(k,:);
                    ind=ind+1;
                end
            end
        end

        fid2_mocor=conj(fid2_mocor);
    end
    % EM2 (end) (16/02/2026)
    
    %% 4-add all the info to the Matlab study structure
    if dosaveprocessing
        load(char(fullfile(folder_results,'raw',filelist(file).name)));
        study.fidaprocess.phsa=phsa;
        study.fidaprocess.fsa=fsa;
        study.fidaprocess.metrica=metrica;
        study.fidaprocess.badAveragesa=badAveragesa; 
        study.fidaprocess.phsb=phsb;
        study.fidaprocess.fsb=fsb;
        study.fidaprocess.metricb=metricb;
        study.fidaprocess.badAveragesb=badAveragesb;

        study.params.nt=size(fidmocor,1)*2; 
        study.multiplicity=size(fidmocor,1);

        study.process.apodparam1=zeros(1,size(fidmocor,1));
        study.process.apodparam2=zeros(1,size(fidmocor,1));
        study.process.phasecorr0=zeros(1,size(fidmocor,1));
        study.process.phasecorr1=zeros(1,size(fidmocor,1));

        study.data.real=zeros(size(fidmocor,1),1,size(fidmocor,2));
        study.data.real(:,1,:)=real(fidmocor);
        study.data.imag=zeros(size(fidmocor,1),1,size(fidmocor,2));
        study.data.imag(:,1,:)=imag(fidmocor);

        % EM2 (16/02/2026)
        % if ~exist([pwd '/processed/'], 'dir') moved to the beginning
        %     mkdir([pwd '/processed/'])
        % end
        
        save(fullfile(folder_results,'processed', [filelist(file).name(1:end-4) '_processed.mat']),'study')

        % EM2 8end)(16/02/2026)

        % EM2 (16/02/2026)
        if job0 == 2
            load(char(fullfile(folder_results,'raw',filelist2(file).name)));
            study.fidaprocess.phsa=phsa2;
            study.fidaprocess.fsa=fsa2;
            study.fidaprocess.metrica=metric2_a;
            study.fidaprocess.badAveragesa=badAverages2_a;
            study.fidaprocess.phsb=phsb2;
            study.fidaprocess.fsb=fsb2;
            study.fidaprocess.metricb=metric2_b;
            study.fidaprocess.badAveragesb=badAverages2_b;

            study.params.nt=size(fid2_mocor,1)*2;
            study.multiplicity=size(fid2_mocor,1);

            study.process.apodparam1=zeros(1,size(fid2_mocor,1));
            study.process.apodparam2=zeros(1,size(fid2_mocor,1));
            study.process.phasecorr0=zeros(1,size(fid2_mocor,1));
            study.process.phasecorr1=zeros(1,size(fid2_mocor,1));

            rmfield(study.data,'real')
            rmfield(study.data,'imag')

            study.data.real=zeros(size(fid2_mocor,1),1,size(fid2_mocor,2));
            study.data.real(:,1,:)=real(fid2_mocor);
            study.data.imag=zeros(size(fid2_mocor,1),1,size(fid2_mocor,2));
            study.data.imag(:,1,:)=imag(fid2_mocor);

            if ~exist(fullfile(folder_results, 'processed'), 'dir')
               mkdir(fullfile(folder_results, 'processed'))
            end
            save(fullfile(folder_results, 'processed', [filelist2(file).name(1:end-4) '_processed_2.mat']),'study')
        end
        % EM2 (end) (16/02/2026)

    end 
    end 
end 
    
%% STEP 2 - APPLY PREPROCESSING, SUM AND PHASE THE SUM 

if dosavesum

screenSz = get(0,'ScreenSize'); figure('Position',[screenSz(3)/5*4,screenSz(4)/3-150,screenSz(3)/5,screenSz(4)/3],'Color','w');

for file=1:length(filelist)
    % EM2 (16/02/2026)
    load(fullfile(folder_results, 'processed',[filelist(file).name(1:end-4) '_processed.mat']),'study')
    % EM2 (end) (16/02/2026)

    fidmocor=squeeze(study.data.real)+1i*squeeze(study.data.imag);
    sumfid=sum(fidmocor);%./(size(fidmocor,1).*2); 
   
    
    %% save
    study.data.real=zeros(1,1,study.np/2);
    study.data.imag=zeros(1,1,study.np/2);

    study.data.real(1,1,:)=real(sumfid); 
    study.data.imag(1,1,:)=imag(sumfid);

    study.multiplicity=1;
    study.process.lsfid=0;
    study.process.apodparam1=0;
    study.process.apodparam2=0;
    study.process.phasecorr0=0;
    study.process.phasecorr1=0;
    study.process.B0=zeros(1,study.np/2);

    filename=filelist(file).name(1:end-4);

    % EM2 (16/02/2026)
    % if ~exist([pwd '/processed/sum/'], 'dir')
    %     mkdir([pwd '/processed/sum/'])
    % end
    % save([pwd '/processed/sum/SUM_' filename '_processed.mat'],'study');
    if ~exist(fullfile(folder_results, 'processed', 'sum'), 'dir')
       mkdir(fullfile(folder_results, 'processed', 'sum'))
    end
    save(fullfile(folder_results, 'processed', 'sum',['SUM_' filename '_processed.mat']),'study');
    % EM2 (end) (16/02/2026)    %plot

    % phi = 0.7; %%PHASE 
    receiveroffset_ppm = 4.7;
    sfrq1H = study.resfreq;
    receiveroffset_hz=receiveroffset_ppm*sfrq1H;
    frqEch = study.spectralwidth;
    np = study.np/2;
    dw=1/frqEch;
    sifactor=1;
    fmax=1/(2*dw);
    f_vec=[-fmax:2*fmax/(sifactor*np-1):fmax];
    ppm_vec=(-f_vec+receiveroffset_hz)/sfrq1H;
    % time=((0:np-1)*dw)';


    % figure; 
    ftcorr=fftshift(fft(sumfid.*exp(-1i*phi)./study.params.nt,[],2),2); 
    plot(ppm_vec,real(ftcorr))
    hold on 
    load(char(fullfile(folder_results,'raw',[filelist(file).name(1:end-4) '.mat'])),'study')   
    fidini=squeeze(study.data.real)+1i*squeeze(study.data.imag);
    sumfidini=sum(fidini)./study.params.nt;
    sumfidini=[sumfidini(77:end),zeros(1,76)];  
    ftini=fftshift(fft(sumfidini.*exp(-1i*phi),[],2),2); 
    plot(ppm_vec,real(ftini)); xlim(gca,[2000 3500]);
    set(gca,'XDir','reverse')
    xlabel('\delta [ppm]')
    legend('Corrected','Original','Box','off','Location','southeast'); xlim tight;  ylim  tight;

    xlim tight; 
    ylim tight;
    
    % Save the plot as PNG
    saveas(gcf, fullfile(folder_results, 'processed', 'sum', ['SUM_' filename '_processed.png'])); 

    % EM2 (16/02/2026)
    if job0 == 2
        load(fullfile(folder_results, 'processed', [filelist2(file).name(1:end-4) '_processed_2.mat']),'study')

        fid2_mocor=squeeze(study.data.real)+1i*squeeze(study.data.imag);
        sumfid2=sum(fid2_mocor);%./(size(fidmocor,1).*2);


        %% save
        study.data.real=zeros(1,1,study.np/2);
        study.data.imag=zeros(1,1,study.np/2);

        study.data.real(1,1,:)=real(sumfid2);
        study.data.imag(1,1,:)=imag(sumfid2);

        study.multiplicity=1;
        study.process.lsfid=0;
        study.process.apodparam1=0;
        study.process.apodparam2=0;
        study.process.phasecorr0=0;
        study.process.phasecorr1=0;
        study.process.B0=zeros(1,study.np/2);

        filename=filelist2(file).name(1:end-4);
        if ~exist(fullfile(folder_results, 'processed'), 'dir')
           mkdir(fullfile(folder_results, 'processed'))
        end
    save(fullfile(folder_results, 'processed', 'sum', ['SUM_' filename '_processed_2.mat']),'study');

    end
    % EM2 (end) (16/02/2026)

end
end 

%disp('**************** done ****************')
toc
end