
function [] = c_Quantif_LCModel_noabsquant_MB(folder_results, scan_id, basis_set, LCMpath)

% from Guillaume's codes

% Define folders
foldersum          = fullfile(folder_results, 'processed','sum');
folderrawsave      = fullfile(folder_results, 'raw');
folder_quanti_save = fullfile(folder_results,'quantified');

doraw=true;
dowritecontrolfile=true;
doLCM=true;


if ~exist(folderrawsave, 'dir')
     mkdir(folderrawsave)
end

LCModel_results_folder = folder_quanti_save;
if ~exist(LCModel_results_folder,'dir')
    mkdir(LCModel_results_folder)
end

filelist = dir(fullfile(foldersum, '*.mat'));
file = filelist(contains({filelist.name}, ['_' num2str(scan_id) '_']));

%% 1- create . RAW from SUM_... .mat
disp(file.name)

load(fullfile(foldersum, file.name));
filenameRAW = file.name(1:end-4);
filenameCONTROL = [filenameRAW,'.CONTROL'];
filename = fullfile(folderrawsave,[filenameRAW,'.RAW']);

%"Normalize as we would do when clicking Normalize in Nico's GUI
norm_factor = study.params.nt;

data.real = squeeze(study.data.real) ./ norm_factor;
data.imag = squeeze(study.data.imag) ./ norm_factor;

sdata = length(data.real);
NFFT = 2^nextpow2((sdata));

RAW.real = zeros([NFFT 1]);
RAW.imag = zeros([NFFT 1]);
RAW.real = data.real; % fill in your data
RAW.imag = data.imag; % fill in your data
    
ID = '';
rawfile = filename;
tramp = 1;
volume = study.params.vox1 .* study.params.vox2 .* study.params.vox3;

%header raw file
fileid = fopen(rawfile, 'w','b');
fprintf(fileid,' \n $NMID\n');
fprintf(fileid,' ID=\''%s\''\n',ID);
fprintf(fileid,' FMTDAT=\''(2E13.5)\''\n');
datastring=strrep(sprintf(' TRAMP= % 13.5E\n', tramp), 'E+0', 'E+');
datastring=strrep(datastring, 'E-0', 'E-');
fprintf(fileid,datastring);
datastring=strrep(sprintf(' VOLUME= % 13.5E\n', volume), 'E+0', 'E+');
datastring=strrep(datastring, 'E-0', 'E-');
fprintf(fileid,datastring);
fprintf(fileid,' $END\n');
%data
for k = 1:1:length(RAW.real)
    datastring = sprintf(' % 13.5E % 13.5E\n',RAW.real(k), RAW.imag(k));
    datastring = strrep(datastring, 'E+0', 'E+');
    datastring = strrep(datastring, 'E-0', 'E-');
    fprintf(fileid,datastring);

end
fclose(fileid);


%% 2-create CONTROL file
close all;
if dowritecontrolfile
    disp ('Creating .CONTROL...')
    
    cfile = fullfile(LCModel_results_folder,filenameCONTROL);
    
    fileid = fopen(cfile,'w');
    fprintf(fileid,' $LCMODL\n');
    fprintf(fileid,' ATTH2O = 1.0\n'); % attenuation of the NMR-visible water signal
    fprintf(fileid,' NCOMBI = 4\n');
    fprintf(fileid," CHCOMB(1) = 'NAA+NAAG'\n");
    fprintf(fileid," CHCOMB(2) = 'Glu+Gln'\n");
    fprintf(fileid," CHCOMB(3) = 'GPC+PCho'\n");
    fprintf(fileid," CHCOMB(4) = 'Cr+PCr'\n");
    fprintf(fileid,'NRATIO = 0\n'); 
    fprintf(fileid,'NSIMUL = 0\n'); 
    fprintf(fileid,'NOMIT = 0\n'); 
    
    USE1 = {'NAA','Cr','PCr','Tau','Gln','Ins','PCho'}; % Basic spectra in primary analysis
    fprintf(fileid,' NUSE1 = %i\n',length(USE1));
    for j = 1:length(USE1)
        fprintf(fileid," CHUSE1(%i) = '%s'\n",j,USE1{j});
    end
    fprintf(fileid,' CONREL = %.2f\n',8.0); % Relative metabolite concentration
    fprintf(fileid,' DELTAT = %.2i\n',1./5000); % 1/samplerate was before 1/7142
    fprintf(fileid,' DKNTMN = %1.2f\n',0.25); % DKNTMN
    fprintf(fileid,' DOECC = F\n'); % DO Eddy current correction
    fprintf(fileid,' DOWS = F\n'); % DO water scaling
    fprintf(fileid,' DOREFS = T\n'); % DO Cr referencing
    
    fprintf(fileid,' FWHMBA = 0.0033\n');
    fprintf(fileid,' HZPPPM = %.3f\n',400.264); % NMRfreq for 14.1 T: 599.419 , for 9.4 T:400.264
    fprintf(fileid,' LCOORD = 9\n'); % Save coord file, Y : LCOORD = 9, N : LCOORD = 0
    fprintf(fileid," NAMREL = '%s'\n",'Cr+PCr'); % Relative metabolite name
    fprintf(fileid,' NCALIB = 0\n');
    fprintf(fileid,' NUNFIL = %i\n',2048); % FidPoints
    fprintf(fileid," PGNORM = 'US'\n");
    fprintf(fileid,' PPMEND = %1.2f\n',0.2); % right edge of the window
    fprintf(fileid,' PPMST = %1.2f\n',4.3); % left edge of the window
    fprintf(fileid,' RFWHM = 1.8\n');
    
    fprintf(fileid,' VITRO = F\n');
    
    fprintf(fileid,' WCONC = %1.2f\n',44444); % the NMR-visible water concentration (mM) in the voxel
    fprintf(fileid,' NEACH = 999\n');
    fprintf(fileid,' SHIFMN = -0.2,-0.1\n');
    fprintf(fileid,' SHIFMX = 0.3,0.3\n');
    fprintf(fileid,' KEY = %i\n',210387309); % Licence KEY
    fprintf(fileid," OWNER = '%s'\n",'Center for Biomedical Imaging, Lausanne'); % Licence OWNER
    fprintf(fileid," FILBAS = '%s'\n",basis_set);
    fprintf(fileid,' DEGPPM = 0\n');
    fprintf(fileid,' DEGZER = 0\n');
    fprintf(fileid,' SDDEGP = 10\n');
    fprintf(fileid,' SDDEGZ = 10\n');
    add_title = char(datetime);
    fprintf(fileid," TITLE = '%s %s'\n",filenameRAW,add_title);
    fprintf(fileid," FILPS = '%s'\n",fullfile(LCModel_results_folder,[filenameRAW,'.ps']));
    fprintf(fileid," FILCOO = '%s'\n",fullfile(LCModel_results_folder,[filenameRAW,'.coord']));
    fprintf(fileid," FILTAB = '%s'\n",fullfile(LCModel_results_folder,[filenameRAW,'.table']));
    fprintf(fileid," FILRAW = '%s'\n",fullfile(folderrawsave,[filenameRAW,'.RAW']));
    fprintf(fileid,' $END');
    fclose(fileid);

end

%% 3- Run LCModel
if doLCM
    disp ('LCModel quantification...')
    Cfile_name=fullfile(LCModel_results_folder,filenameCONTROL);
    cd(LCMpath)
    [status, cmdout] = system(['./lcmodel < "' Cfile_name '"']);
    cd(folder_results)
end 


end
