function metadata = parseImgs(src)
% PARSEIMGS Parse image filenames.
%
% PARSEIMGS(src)
%     If src is a folder (char or string scalar), returns a structure array
%     of metadata of all .b16 files in that folder, renaming any files that
%     lack a five-digit _SEQ suffix to _00001.b16.
%
%     If src is a list of existing file-paths (string array or cellstr),
%     the function filters that list rather than scanning a folder.
%
% Returns:
%     metadata - structure array of matadata
%
% See also FILEPARTS, INPUTPARSER, REGEXP, MOVEFILE.

% Author: Xukun Lin
% Date: 05.24.2025

    if isempty(src)
        metadata = [];
        return;
    end

    % Convert all strings to character arrays
    src = convertContainedStringsToChars(src);

    % Determine whether the input is a folder or files
    srcIsFolder = numel(isfolder(src)) == 1 && isfolder(src);
    srcIsFiles = all(isfile(src));

    % Extract filenames
    if srcIsFolder                                   % the input is a folder
        names = {dir(fullfile(src, '*.b16')).name};
    elseif srcIsFiles                                % the input are files
        temp = convertContainedStringsToChars(src);
        [fs, ns, es] = fileparts(temp);
        names = strcat(ns, es);
    else
        error('The input is invalid.')
    end

    % Rename files without _SEQ
    for i = 1:numel(names)
        name = names{i};
        if isempty(regexp(name, '_\d{5}\.b16$', 'once'))
            base = name(1:(end - 4));
            newname = sprintf('%s_%05d.b16', base, 1);
            if srcIsFolder
                movefile(fullfile(src, name), fullfile(src, newname));
            else
                names{i} = newname;
            end
        end
    end

    % Reload filenames if the input is a folder
    if srcIsFolder
        names = {dir(fullfile(src, '*.b16')).name};
    end

    % Parser for filenames
    expr = ['^', ...
        '(?<type>F2|F1|BG|FL|DK)_', ...         % type
        '(?<curr>-?\d+)(?<cunit>uA|mA|A)_', ... % current + unit
        '(?<int>\d+)(?<iunit>us|ms|s)_', ...    % integration time + unit
        '(?<bin>\d+x\d+)_', ...                 % bin = “Nx x Ny”
        '(?<seq>\d{5})', ...                    % seq exactly 5 digits
        '\.b16$'];

    % Parse and store information in a structure array
    metadata = struct('Filename', {}, 'Type', {}, 'Curr', {}, 'Tint', {}, 'BinX', {}, 'BinY', {}, 'Seq', {});
    
    for i = 1:numel(names)
        name = names{i};
        tk = regexp(name, expr, 'names');
        if isempty(tk)
            warning('Skipping unrecognized filename: %s', name);
            continue;
        end

        % Convert to A
        switch tk.cunit
            case 'uA', cfactor = 1e-6;
            case 'mA', cfactor = 1e-3;
            case 'A',  cfactor = 1;
        end
        I = str2double(tk.curr) * cfactor;

        % Convert to second
        switch tk.iunit
            case 'us', ifactor = 1e-6;
            case 'ms', ifactor = 1e-3;
            case 's',  ifactor = 1;
        end

        Tint = str2double(tk.int) * ifactor;
        bin  = sscanf(tk.bin, '%dx%d');
        seq  = str2double(tk.seq);

        if srcIsFolder
            metadata(i).Filename = fullfile(src, name);
        else
            metadata(i).Filename = fullfile(fs(i), name);
        end

        metadata(i).Type = tk.type;
        metadata(i).Curr = I;
        metadata(i).Tint = Tint;
        metadata(i).BinX = bin(1);
        metadata(i).BinY = bin(2);
        metadata(i).Seq  = seq;
    end
end