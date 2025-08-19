function metadataf = filterImgs(metadata, varargin)
% FILTERIMGS Filter images.
%
% FILTERIMGS(metadata, Name, Value, ...)
%     Filters by information encoded in metadata. Supported:
%       'Type': char, string, string array, cellstr, or cell of strings
%       'Curr': numeric vector (in amperes)
%       'Tint': numeric vector (in seconds)
%       'Bin' : Nx2 numeric matrix of [binX, binY] (allowed bins: 1, 2, 4)
%       'Seq' : numeric vector of sequence indices
%
%     Required fields in metadata: 'Type', 'Tint', 'BinX', 'BinY'.
%
% Returns:
%     metadataf - filtered struct array
% 
% Example:
%     metadata = parseImgs('C:/data');
%     signalImgs = filterImgs(metadata, 'Type', {'F1', 'F2'});

% Author: Xukun Lin
% Date:   05.24.2025
    
    if isempty(metadata)
        metadataf = [];
        return;
    end

    % Convert all strings to character arrays
    varargin = convertContainedStringsToChars(varargin);

    % Parse input (case insensitive)
    p = inputParser;
    p.FunctionName = mfilename;
    isValidsrc = @(x) isstruct(x) && all(isfield(x, {'Type', 'Tint', 'BinX', 'BinY'}));
    addRequired(p, 'Metadata', isValidsrc);
    addParameter(p, 'Type', [], @(x) ischar(x) || isstring(x) || iscellstr(x));
    addParameter(p, 'Curr', [], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'Tint', [], @(x) isnumeric(x) && isvector(x));
    addParameter(p, 'Bin',  [], @(x) size(x, 2) == 2 && all(ismember(unique(x), [1, 2, 4])));
    addParameter(p, 'Seq',  [], @(x) isnumeric(x) && isvector(x));
    parse(p, metadata, varargin{:});
    params = p.Results;

    % Apply filter
    keep = true(1, numel(params.Metadata));

    if ~isempty(params.Type)
        T = cellstr(params.Type);
        keep = keep & ismember({params.Metadata.Type}, T);
    end
    if ~isempty(params.Curr)
        if ~isfield(params.Metadata, 'Curr')
            warning('The input metadata struct does not have field Curr. Ignoring the Curr filter.')
        else
            keep = keep & ismember([params.Metadata.Curr], params.Curr);
        end
    end
    if ~isempty(params.Tint)
        keep = keep & ismember([params.Metadata.Tint], params.Tint);
    end
    if ~isempty(params.Bin)
        b = [[params.Metadata.BinX]', [params.Metadata.BinY]'];
        keep = keep & ismember(b, params.Bin, 'rows')';
    end
    if ~isempty(params.Seq)
        if ~isfield(params.Metadata, 'Seq')
            warning('The input metadata struct does not have field Seq. Ignoring the Seq filter.')
        else
            keep = keep & ismember([params.Metadata.Seq], params.Seq);
        end
    end

    metadataf = params.Metadata(keep);
end