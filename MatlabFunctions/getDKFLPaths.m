function [DKPath, FLPath] = getDKFLPaths(exposure, batchTag)
% GETDKFLPATHS  Return DK/FL paths for a given exposure, relative to project root.
%
% Folder layout:
%   project/
%     MatlabCodes/      % <-- this file lives here
%     <batch>_FL/
%     <batch>_DK/
%
% Usage:
%   [DK, FL] = getDKFLPaths(0.2, '20250525');
%   [DK, FL] = getDKFLPaths(2);              % defaults to '20250525'
%
% Supports exposure = 0.2 (-> '200ms') or 2 (-> '2s').
%
% Author: Kelvin Titimbo | Date: 20250815

    if nargin < 2 || isempty(batchTag)
        batchTag = '20250525';  % default batch folder prefix
    end

    % --- map exposure to subfolder name
    estring = exposure_to_str(exposure);

    % --- locate project root (parent of MatlabCodes)
    thisDir     = fileparts(mfilename('fullpath'));  % .../project/MatlabCodes
    projectRoot = fileparts(thisDir);                % .../project

    % --- compose base folders
    FLBase = fullfile(projectRoot, [batchTag, '_FL']);
    DKBase = fullfile(projectRoot, [batchTag, '_DK']);

    % --- include exposure subfolder if present (e.g., 200ms, 2s)
    FLPath = fullfile(FLBase, estring);
    DKPath = fullfile(DKBase, estring);

    % fallback: if exposure subfolder doesn't exist, use base
    if ~isfolder(FLPath), FLPath = FLBase; end
    if ~isfolder(DKPath), DKPath = DKBase; end

    % sanity checks
    assert(isfolder(FLPath), 'FL path not found: %s', FLPath);
    assert(isfolder(DKPath), 'DK path not found: %s', DKPath);
end

function es = exposure_to_str(exposure)
    % normalize exposure input to folder name
    if isnumeric(exposure)
        if abs(exposure - 0.2) < 1e-12
            es = '200ms';
            return
        elseif abs(exposure - 2) < 1e-12
            es = '2s';
            return
        end
    elseif isstring(exposure) || ischar(exposure)
        es = char(exposure);  % allow passing '200ms' or '2s' directly
        return
    end
    error('Unsupported exposure: %s', mat2str(exposure));
end