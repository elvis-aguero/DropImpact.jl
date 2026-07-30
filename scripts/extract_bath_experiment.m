% Extract the BATH EXPERIMENTAL rebound metrics from the source .fig files of
% Alventosa, Cimpeanu & Harris, "Inertio-capillary rebound of a droplet impacting a fluid
% bath", JFM (2023), doi:10.1017/jfm.2023.88.
%
% WHY THIS EXISTS. The experimental points are stored in the figures as `errorbar` objects,
% one object per data point. A CSV export that walks only `line` children silently drops every
% one of them -- which is what happened in
% km-dropplet-onto-bath/.../water_QPExp_DNS_3panel_FFF_extracted/, whose CSVs contain the eight
% model curves and none of the five experimental points. This reads the errorbar objects
% directly, so the values are the authors' own numbers rather than a digitisation of a raster.
%
% METRIC DEFINITIONS, which are NOT the same for experiment and model (paper, sec. 4.2):
%   experiment : t_c is the interval between the two instants the NORTH POLE crosses z = 2R,
%                and alpha is minus the ratio of the vertical velocities at those instants;
%                the detachment instant was not optically resolvable.
%   model/DNS  : the same construction but on the CENTRE OF MASS crossing z = R, chosen
%                because it better reflects translational energy transfer.
% The authors quote a typical difference between the two conventions of 5% for the deionized
% water experiments and 2% for the silicone oil. Our threshold_contact_time implements the
% MODEL convention (z_cm < R), so a residual of that order against these columns is expected
% from the definition alone and must not be read as model error.
%
% Run:  matlab -nodisplay -batch "run('scripts/extract_bath_experiment.m')"
% Writes: data/experiments/bath_experiment_<fluid>.csv

FIGDIR = fullfile(getenv('HOME'), 'Documents/Github/km-dropplet-onto-bath/matlab/1_code/Figures');
HERE   = fileparts(mfilename('fullpath'));
OUTDIR = fullfile(HERE, '..', 'data', 'experiments');

% fluid, figure file, Bo, Oh  (Bo and Oh from the figure captions, not from the .fig)
cases = { 'water', 'water_QPExp_DNS_3panel_FFF.fig', 0.017, 0.006 ; ...
          'oil',   'oil_3panel_FINALF.fig',          0.056, 0.058 };

for c = 1:size(cases, 1)
    fluid = cases{c,1};  Bo = cases{c,3};  Oh = cases{c,4};
    fpath = fullfile(FIGDIR, cases{c,2});
    if ~isfile(fpath)
        fprintf('SKIP %s: %s not found\n', fluid, fpath); continue
    end
    f = openfig(fpath, 'invisible');
    ax = flipud(findall(f, 'type', 'axes'));      % panel order (a), (b), (c)

    % Identify panels by their y-label rather than by index, so a reordered figure cannot
    % silently swap contact time for penetration depth.
    metric = cell(numel(ax), 1);
    for i = 1:numel(ax)
        yl = get(get(ax(i), 'YLabel'), 'String');
        if ischar(yl) || isstring(yl), yl = char(yl); else, yl = char(strjoin(string(yl))); end
        if     contains(yl, 't_c'),    metric{i} = 'tc';
        elseif contains(yl, 'alpha'),  metric{i} = 'cor';
        elseif contains(yl, 'delta'),  metric{i} = 'delta';
        else,                          metric{i} = sprintf('axes%d', i);
        end
    end

    % Collect every errorbar point of every panel, keyed by We.
    M = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for i = 1:numel(ax)
        eb = findall(ax(i), 'type', 'errorbar');
        for j = 1:numel(eb)
            x = get(eb(j), 'XData');  y = get(eb(j), 'YData');
            lo = get(eb(j), 'YNegativeDelta');  hi = get(eb(j), 'YPositiveDelta');
            if isempty(lo), lo = NaN; end
            if isempty(hi), hi = NaN; end
            for k = 1:numel(x)
                key = sprintf('%.6f', x(k));
                if ~isKey(M, key), M(key) = struct('We', x(k)); end
                s = M(key);
                s.(metric{i})            = y(k);
                s.([metric{i} '_sd'])    = mean([lo(min(k,end)), hi(min(k,end))]);
                M(key) = s;
            end
        end
    end
    close(f);

    keys_ = keys(M);
    we = cellfun(@(k) M(k).We, keys_);
    [~, ord] = sort(we);
    cols = {'tc', 'cor', 'delta'};

    out = fullfile(OUTDIR, sprintf('bath_experiment_%s.csv', fluid));
    fid = fopen(out, 'w');
    fprintf(fid, 'We,tc_over_tsigma,tc_sd,cor,cor_sd,delta_over_R,delta_sd,Bo,Oh\n');
    for idx = ord
        s = M(keys_{idx});
        v = zeros(1, 6);
        for m = 1:3
            v(2*m-1) = getfielddef(s, cols{m});
            v(2*m)   = getfielddef(s, [cols{m} '_sd']);
        end
        fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n', s.We, v, Bo, Oh);
    end
    fclose(fid);
    fprintf('wrote %s : %d experimental points, Bo = %.3f, Oh = %.3f\n', ...
            out, numel(ord), Bo, Oh);
end

function v = getfielddef(s, f)
    if isfield(s, f), v = s.(f); else, v = NaN; end
end
