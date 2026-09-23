function processing_dmrs_matlab(input_path, output_path, data_folders_list, coil_type, basis_set, LCMpath, job0)

% Needed for compilation
set(0,'DefaultFigureVisible','off');
warning('off','all');

% Loop through data folders
data_folders_list = str2num(data_folders_list);
job0 = str2num(job0);

for j = 1:length(data_folders_list)
        % Get scan id
        scan_id = data_folders_list(j);
        [~, raw_data,~] = fileparts(input_path);
        
        fprintf('\n');
        disp('==============================================================');
        fprintf(' Processing dataset: %s\n', raw_data);
        fprintf(' Scan number       : %d\n', scan_id);
        disp('==============================================================');

        % ------------------------------------------------------------------
        % Import data from Bruker
        fprintf('\n[1/3] Importing Bruker data\n');
        fprintf('      with a_Create_study_Bruker_RatCryo_9p4T_EM_MB\n');
        fprintf('        (Toi version 10.11.2025)\n\n');
    
        a_Create_study_Bruker_RatCryo_9p4T_EM_MB( ...
            input_path, output_path, scan_id, coil_type);

        % ------------------------------------------------------------------
        % Preprocessing
        fprintf('\n[2/3] Preprocessing data\n');
        fprintf('      with b_Preprocessing_FidA_9p4T_EM_MB\n');
        fprintf('        (Toi version 10.11.2025)\n\n');
    
        phi = 5.6;  % phase correction, only for visualization purposes
        b_Preprocessing_FidA_9p4T_EM_MB(output_path, scan_id, phi, job0);
        close all

        % ------------------------------------------------------------------
        % Quantification
        fprintf('\n[3/3] Quantifying spectra with LCModel \n');
        fprintf('      with c_Quantif_LCModel_noabsquant_MB\n');
        fprintf('        (adapted by MB in Jan 2026)\n\n');
        c_Quantif_LCModel_noabsquant_MB(output_path, scan_id, basis_set, LCMpath)
    
        fprintf('\n ✔ Finished scan %d\n', scan_id);
        disp('--------------------------------------------------------------');
    
end

end