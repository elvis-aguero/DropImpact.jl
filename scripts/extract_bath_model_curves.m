% Extract the MODEL CURVES (1PKM quasi-potential, and DNS) from the contact-time panel of the
% bath figures of AlventosaEtAl2023, so this package's residual against experiment can be set
% beside the published model's residual against the SAME experimental points.
%
% Per the figure captions: "Predictions of the quasi-potential model are shown as blue solid
% lines, DNS as black dashed lines." Curves are `line` children; the experimental points are
% `errorbar` children and are handled by extract_bath_experiment.m instead.
%
% ALL THREE PANELS are extracted, not only contact time: the coefficient of restitution and the
% maximum penetration depth matter too, and delta in particular, because the paper reports that its
% own model underpredicts BOTH tc and delta at intermediate We. Without the published delta curve
% there is no way to tell whether a delta deficit in this package is a defect of this package or a
% property of the whole reduced-model class.
%
% Run:  matlab -nodisplay -batch "run('scripts/extract_bath_model_curves.m')"
% Writes: data/experiments/bath_model_curves_<fluid>.csv  (one row per curve vertex)

FIGDIR = fullfile(getenv('HOME'), 'Documents/Github/km-dropplet-onto-bath/matlab/1_code/Figures');
HERE   = fileparts(mfilename('fullpath'));
OUTDIR = fullfile(HERE, '..', 'data', 'experiments');

cases = { 'water', 'water_QPExp_DNS_3panel_FFF.fig' ; ...
          'oil',   'oil_3panel_FINALF.fig'          };

for c = 1:size(cases, 1)
    fluid = cases{c,1};
    fpath = fullfile(FIGDIR, cases{c,2});
    if ~isfile(fpath), fprintf('SKIP %s: %s not found\n', fluid, fpath); continue, end
    f = openfig(fpath, 'invisible');
    ax = flipud(findall(f, 'type', 'axes'));

    out = fullfile(OUTDIR, sprintf('bath_model_curves_%s.csv', fluid));
    fid = fopen(out, 'w');
    fprintf(fid, 'metric,curve,linestyle,r,g,b,npts,We,value\n');
    for i = 1:numel(ax)
        % Identify each panel by its y-label, not by index.
        yl = get(get(ax(i), 'YLabel'), 'String');
        if ischar(yl) || isstring(yl), yl = char(yl); else, yl = char(strjoin(string(yl))); end
        if     contains(yl, 't_c'),   metric = 'tc';
        elseif contains(yl, 'alpha'), metric = 'cor';
        elseif contains(yl, 'delta'), metric = 'delta';
        else,  fprintf('%s: skipping unrecognised panel "%s"\n', fluid, yl); continue
        end
        ln = findall(ax(i), 'type', 'line');
        for j = 1:numel(ln)
            x = get(ln(j), 'XData'); y = get(ln(j), 'YData');
            col = get(ln(j), 'Color'); sty = get(ln(j), 'LineStyle');
            if numel(x) < 3, continue, end          % markers and guides, not curves
            for k = 1:numel(x)
                fprintf(fid, '%s,%d,%s,%.4f,%.4f,%.4f,%d,%.8f,%.8f\n', ...
                        metric, j, sty, col(1), col(2), col(3), numel(x), x(k), y(k));
            end
            fprintf('%s %-5s curve %d: style=%-4s color=[%.2f %.2f %.2f] npts=%d We=[%.4f..%.4f] val=[%.4f..%.4f]\n', ...
                    fluid, metric, j, sty, col(1), col(2), col(3), numel(x), ...
                    min(x), max(x), min(y), max(y));
        end
    end
    fclose(fid);
    close(f);
    fprintf('wrote %s\n\n', out);
end
