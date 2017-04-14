function bbs = my_edgeboxes(image_file_name_bbs, model, varargin)
    
    prev_image_name = 'a';
    image_name = 'b';

    [filepath, ~, ~] = fileparts(image_file_name_bbs);

    while 1,
        pause(0.1);

        aux = 1;
        try
            f = fopen(image_file_name_bbs, 'r');
            image_name = fgetl(f);
        catch ME
            if strcmp(ME.identifier, 'MATLAB:FileIO:InvalidFid'),
                aux = 0;
                continue;
            end;
        end;

        full_name = [filepath '/' image_name]; 

        if aux == 1 & ~strcmp(image_name, prev_image_name) & exist(full_name) == 2,
            bbs = edgeBoxes([filepath '/' image_name], model, varargin);
            prev_image_name = image_name;

            dlmwrite([filepath '/' image_name(1:end-4) '_bbs.txt'], bbs);
        else
            fclose('all');
        end
            
    end
end
