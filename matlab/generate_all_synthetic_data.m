function [ ] = generate_all_synthetic_data(batterySOCs )
%GENERATE_ALL_SYNTHETIC_DATA generate synthetic data for specified initial
%SOCs
% get_param('battery_synthetic/battery','ObjectParameters')
%   
    
    saveRoot = './synthetic_data/';
    if not(exist(saveRoot, 'dir'))
        mkdir(saveRoot);
    end
    mdl = 'battery_synthetic';
    in = Simulink.SimulationInput(mdl);
    root = './exportEpisodes';
    %root = './test';
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
        
        maxQ = 386;
        ndC = 68;
        cnv = 326;
        expCurrent = 95;
        for k = 1:length(batterySOCs)
            
            agedMaxQ = num2str( int32(maxQ * batterySOCs(k) / 100));
            agedNdc = num2str(int32(ndC * batterySOCs(k) / 100));
            agedCnv = num2str(int32(cnv * batterySOCs(k) / 100));
            agedExpC = num2str(int32(expCurrent * batterySOCs(k) / 100));
            
            set_param( strcat( mdl,'/battery'),'expZone',strcat("[25.6094    ",agedExpC,"]"));
            
            set_param( strcat( mdl,'/battery'),'MaxQ',agedMaxQ);
            set_param( strcat( mdl,'/battery'),'Dis_rate',agedNdc);
            set_param( strcat( mdl,'/battery'),'Normal_OP',agedCnv);
            
            agedQ = int32(340 * batterySOCs(k) / 100);
            nomQ2set = num2str(agedQ);
            set_param( strcat( mdl,'/battery'),'NomQ',nomQ2set);
            %set_param( strcat( mdl,'/battery'),'R','0,00070588' )
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