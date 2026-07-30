% Run the reference implementation at OUR canonical reference impact and dump a trace.
% We = 1.0958, Bo = 0.0166, Oh = 0.0062  <=>  R = 0.35 mm water, U = 0.4759 m/s.
rho=998.2; sig=72.2e-3; mu=0.9793e-3; g=9.81;
rho_s=rho; sig_s=sig; mu_s=mu; nu=mu/rho; nu_s=mu_s/rho_s;
R = 0.035e-2;
phys=[R sig rho nu sig_s rho_s nu_s];
b = 25*R; l = 0;
u = besselzero(0,1000,1)'/b;
k = besselzero(-1,1000,1)'/b;
nm = 151; mm = 55; ptype='poly6';
U_in = 0.4759;  tf = 0.02;

tsig = sqrt(rho*R^3/sig);
We = rho*U_in^2*R/sig;  Bo = rho*g*R^2/sig;  Oh = mu/sqrt(rho*sig*R);
fprintf('We = %.4f   Bo = %.5f   Oh = %.5f   t_sigma = %.4e s\n', We, Bo, Oh, tsig);

[t,Y,out,rc] = bounce_alventosa_bessel_implicit(-U_in,tf,u,k,nm,b,l,phys,mm,ptype,g);

fprintf('sizes: numel(t)=%d  size(Y)=[%d %d]  numel(rc)=%d\n', numel(t), size(Y,1), size(Y,2), numel(rc));
% rc and Y/t can differ in length (rc is filled inside the contact branch only), so align to the
% shortest. Truncating rather than padding, so no fabricated samples enter the comparison.
n = min([numel(t), size(Y,1), numel(rc)]);
t = t(1:n); Y = Y(1:n,:); rc = rc(1:n);
fprintf('aligned to n = %d\n', n);

% Y layout: 1=z_cm, 2=v, 3..mm+1 = drop modes, mm+2..2mm = drop mode rates,
%           2mm+1..nm+2mm = bath modes, ..., end = force
zc = Y(:,1)/R;                  % CoM height in units of R
vv = Y(:,2)/U_in;               % velocity normalised by impact speed
ff = Y(:,end);                  % force (dimensional)
tt = t/tsig;                    % time in t_sigma
rcn = rc/R;                     % contact radius in R

% first few drop and bath mode amplitudes, nondimensionalised by R
b2 = Y(:,3)/R; b3 = Y(:,4)/R; b4 = Y(:,5)/R;
a1 = Y(:,2*mm+1)/R; a2 = Y(:,2*mm+2)/R; a3 = Y(:,2*mm+3)/R;

T = table(tt, zc, vv, ff, rcn, b2, b3, b4, a1, a2, a3);
writetable(T, 'reference_trace.csv');
fprintf('wrote reference_trace.csv with %d rows\n', height(T));

% their own metrics, computed as in the driver
ic = find(ff > 0);
if ~isempty(ic)
  fprintf('contact (f>0): t = %.4f .. %.4f  => duration %.4f t_sigma\n', ...
          tt(ic(1)), tt(ic(end)), tt(ic(end))-tt(ic(1)));
end
fprintf('max |rc|/R = %.4f at t = %.4f\n', max(rcn), tt(find(rcn==max(rcn),1)));
fprintf('min z_cm/R = %.4f\n', min(zc));
% CoM-below-R threshold contact time, matching threshold_contact_time
below = find(zc < 1);
if ~isempty(below)
  i0 = below(1); i1 = below(end);
  fprintf('z_cm below R: t = %.4f .. %.4f  => tc(threshold) = %.4f t_sigma\n', ...
          tt(i0), tt(i1), tt(i1)-tt(i0));
end
