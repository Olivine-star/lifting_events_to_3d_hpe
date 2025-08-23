function y = nanmean(x, dim)
% 简化版 nanmean 兼容：使用 mean(...,'omitnan')
    if nargin < 2
        % 默认维度与老 nanmean 一致（第一非单例维）
        dim = find(size(x) ~= 1, 1);
        if isempty(dim), dim = 1; end
    end
    y = mean(x, dim, 'omitnan');
end
