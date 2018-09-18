function [ ] = generate_all_synthetic_data(batterySOCs )
%GENERATE_ALL_SYNTHETIC_DATA generate synthetic data for specified initial
%SOCs
%   
    
    saveRoot = './synthetic_data/';
    if not(exist(saveRoot, 'dir'))
        mkdir(saveRoot);
    end
    mdl = 'battery_synthetic';
    in = Simulink.SimulationInput(mdl);
    root = './exportEpisodes';
    batteryDirs = dir(root);
    
    dirFlags = [batteryDirs.isdir];
    batteryDirs = batteryDirs(dirFlags);
    
    numBatt = size(batteryDirs);
    numBattery = numBatt(1);
    f = waitbar(0,'Synthetize batteries');
    for d = 1:numBattery
        perc = d/numBattery;
        
        batteryName = batteryDirs(d).name;
        
        waitbar(perc,f,strcat('Synthetizing ',batteryName));
        
        episodesPath = strcat(root,'/',batteryName,'/*.csv');
        episodes = dir(episodesPath);
        numEp = size(episodes);
        numEpisode = numEp(1);
        
        for k = 1:length(batterySOCs)
            
            agedQ = int32(340 * batterySOCs(k) / 100);
            nomQ2set = num2str(agedQ);
            set_param( strcat( mdl,'/battery'),'NomQ',nomQ2set);
         
            episodeSaveFolder = strcat(saveRoot,batteryName,'_',num2str( batterySOCs(k)));
            if not(exist(episodeSaveFolder, 'dir'))
                mkdir(episodeSaveFolder);
            end
            for e = 1 : numEpisode
                
                episodePath = strcat(root,'/',batteryName);
                % ts is required in simulink model
                [ts,stopTime] = load_real_data(episodePath,episodes(e).name); 
                in = in.setVariable('ts',ts);
                set_param(mdl, 'StopTime',num2str(stopTime) );
                
                simOut = sim(in);
                fileForSave = strcat(episodeSaveFolder,'/',episodes(e).name);
                csvwrite(fileForSave,[simOut.current.Data  ,simOut.voltage.Data]);
            end
        end
    end
    close(f)
end