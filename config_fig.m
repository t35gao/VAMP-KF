function [cf, ca] = config_fig(fig_pos)
cf = gcf;
cf.WindowStyle = 'normal';
cf.Position = fig_pos;

ca = gca;
ca.FontSize = 20;
end

