function infosSorted = appParse(dirPath)
%PARSEIMAGEFILES  Find and parse image‐filename parameters in a folder
%   infos = parseImageFiles(dirPath) looks for files matching
%     Cur*_*_Bin*_Exp*_Ran*.mat
%   in the directory dirPath (defaults to pwd), and returns a struct array
%   with fields:
%     Filename     (string) full path to each .mat file
%     Current_mA   (double) current in mA
%     BinX         (double) X binning
%     BinY         (double) Y binning
%     Exposure_ms  (double) exposure in ms
%     Range_mA     (double) ammeter range in mA

    if nargin<1 || isempty(dirPath)
        dirPath = pwd;
    end

    % find all .mat files with your pattern
    files = dir(fullfile(dirPath,'Cur*_Bin*_Exp*_Ran*.mat'));
    n     = numel(files);
    
    % preallocate
    infos = struct( ...
      'Filename', cell(n,1), ...
      'Current_mA', cell(n,1), ...
      'BinX', cell(n,1), ...
      'BinY', cell(n,1), ...
      'Exposure_ms', cell(n,1), ...
      'Range_mA', cell(n,1) ...
    );
    
    % regex over the basename (no extension)
    expr = [ ...
      '^Cur(?<CurValue>-?\d+)(?<CurUnit>mA|uA)' ...
      '_Bin(?<BinX>\d+)x(?<BinY>\d+)'            ...
      '_Exp(?<ExpValue>\d+)(?<ExpUnit>ms|us)'    ...
      '_Ran(?<Range>\d+)mA$'                     ...
    ];
    
    for k = 1:n
        name     = files(k).name;
        [~, fname] = fileparts(name);
        toks     = regexp(fname, expr, 'names');
        if isempty(toks)
            warning('Skipping "%s": doesn''t match expected pattern.', name);
            continue;
        end
        
        % full path
        infos(k).Filename = fullfile(dirPath, name);
        
        % parse & convert units
        % current: uA→mA or mA→mA
        unitFactorCur = strcmp(toks.CurUnit,'uA')*1e-3 + strcmp(toks.CurUnit,'mA');
        infos(k).Current_mA = str2double(toks.CurValue) * unitFactorCur;
        
        % binning
        infos(k).BinX = str2double(toks.BinX);
        infos(k).BinY = str2double(toks.BinY);
        
        % exposure: us→ms or ms→ms
        unitFactorExp = strcmp(toks.ExpUnit,'us')*1e-3 + strcmp(toks.ExpUnit,'ms');
        infos(k).Exposure_ms = str2double(toks.ExpValue) * unitFactorExp;
        
        % range in mA
        infos(k).Range_mA = str2double(toks.Range);
    end
    [~, order]   = sort([infos.Current_mA]);   % ascending order
    infosSorted  = infos(order);
end
