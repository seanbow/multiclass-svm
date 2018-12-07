function plot_binary_svm(SVM, X, Y)

[m,N] = size(X);

k = 50; % number of grid subdivisions

if N == 3
    % 3D case, plane.

    cubeXMin = min(X(:,1));
    cubeYMin = min(X(:,2));
    cubeZMin = min(X(:,3));

    cubeXMax = max(X(:,1));
    cubeYMax = max(X(:,2));
    cubeZMax = max(X(:,3));
    
    stepx = (cubeXMax-cubeXMin)/(k-1);
    stepy = (cubeYMax-cubeYMin)/(k-1);
    stepz = (cubeZMax-cubeZMin)/(k-1);
    [x, y, z] = meshgrid(cubeXMin:stepx:cubeXMax,cubeYMin:stepy:cubeYMax,cubeZMin:stepz:cubeZMax);
    mm = size(x);
    x = x(:);
    y = y(:);
    z = z(:);

%     classifier_fn_linear = @(x) SVM.w' * x' + SVM.b;
    f = feval(SVM.predict, [x y z]);
    f = f(:);

    t = Y==1;
    figure
    plot3(X(t, 1), X(t, 2), X(t, 3), 'b.', 'MarkerSize', 20);
    hold on
    plot3(X(~t, 1), X(~t, 2), X(~t, 3), 'r.', 'MarkerSize', 20);
    hold on

    sv = SVM.unnormalize(SVM.svs);
    plot3(sv(:, 1), sv(:, 2), sv(:, 3), 'go');

    x0 = reshape(x, mm);
    y0 = reshape(y, mm);
    z0 = reshape(z, mm);
    v0 = reshape(f, mm);

    [faces,verts] = isosurface(x0, y0, z0, v0, 0, x0);
%     patch('Vertices', verts, 'Faces', faces, 'FaceColor','k','edgecolor', 'none', 'FaceAlpha', 0.5);
    patch('Vertices', verts, 'Faces', faces, 'FaceAlpha', 0.5);
    legend('Y = 1','Y = -1','support vectors','Decision boundary')
    grid on
    box on
    view(3)
    hold off
elseif N == 2
    % 2D case, a line

    cubeXMin = min(X(:,1));
    cubeYMin = min(X(:,2));

    cubeXMax = max(X(:,1));
    cubeYMax = max(X(:,2));
    
    stepx = (cubeXMax-cubeXMin)/(k-1);
    stepy = (cubeYMax-cubeYMin)/(k-1);
    [x, y] = meshgrid(cubeXMin:stepx:cubeXMax,cubeYMin:stepy:cubeYMax);
    mm = size(x);
    x = x(:);
    y = y(:);

    f = feval(SVM.predict, [x y]);
    f = f(:);

    figure; hold on;
    gscatter(X(:,1), X(:,2), Y);

    sv = SVM.unnormalize(SVM.svs);
    plot(sv(:, 1), sv(:, 2), 'go', 'MarkerSize', 10);

    x0 = reshape(x, mm);
    y0 = reshape(y, mm);
    v0 = reshape(f, mm);
    
    contour(x0, y0, v0, [0 0]);
    legend('Y = 1','Y = -1','support vectors','Decision boundary')
    grid on
    hold off
end