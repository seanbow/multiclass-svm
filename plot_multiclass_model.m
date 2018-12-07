function h = plot_multiclass_model(model, X, Y, varargin)

p = inputParser;
p.addParameter('legend', 1);

parse(p, varargin{:});

[m,N] = size(X);

% model = model.trained;

% Choose colors
cmap = hsv(model.K);

if N == 3
    % 3D case, a surface in 3D.

    k = 50; % number of grid subdivisions

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
    
    h = figure; hold on;
    
    for k = 1:model.K
        t = Y==k;
        plot3(X(t, 1), X(t, 2), X(t, 3), '.', 'MarkerSize', 20);
    end
    
    x0 = reshape(x, mm);
    y0 = reshape(y, mm);
    z0 = reshape(z, mm);
    
    % Plot the separating boundary for *each* class!
    
    preds = model.predict([x y z]);
    v0 = reshape(preds, mm);
    
    for k = 1:model.K
        [faces,verts] = isosurface(x0, y0, z0, v0, k, x0);
        patch('Vertices', verts, 'Faces', faces, 'FaceAlpha', 0.5);
    end
    
    grid on
    box on
    view(3)
    hold off
    
elseif N == 2
    % 2D case, lines in 2d

    k = 100; % number of grid subdivisions

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
    
    h = figure; hold on;
    
    preds = model.predict([x y]);

    x0 = reshape(x, mm);
    y0 = reshape(y, mm);
    v0 = reshape(preds, mm);

%     contourf(x0, y0, v0, 1:model.K);
    regions = pcolor(x0, y0, v0);
    set(regions,'EdgeAlpha',0,'FaceAlpha',0.2);
    colormap(cmap);
    
    if strcmp(model.type, "AVA")
        for k1 = 1:model.K
            for k2 = k1 + 1:model.K
                sv = model.unnormalize( model.binary_svms{k1}{k2}.svs );
                plot(sv(:,1), sv(:,2), 'go', 'MarkerSize', 10);
            end
        end
    elseif strcmp(model.type, "OVA")
        for k=1:model.K
            sv = model.unnormalize( model.binary_svms{k}.svs );
            plot(sv(:,1), sv(:,2), 'go', 'MarkerSize', 10);
        end
    else
        sv = model.unnormalize( model.svs );
        plot(sv(:, 1), sv(:, 2), 'go', 'MarkerSize', 10);
    end

    gscatter(X(:,1), X(:,2), Y, cmap);
    
   
%     legend
    grid on
%     hold off
    
    axis([cubeXMin cubeXMax cubeYMin cubeYMax])
end

if ~p.Results.legend
    legend('off');
end
