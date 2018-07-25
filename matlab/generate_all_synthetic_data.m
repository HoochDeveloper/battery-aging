function [ ] = generate_all_synthetic_data(batterySOCs )
%GENERATE_ALL_SYNTHETIC_DATA generate synthetic data for specified initial
%SOCs
%   
    warning off;
    files = dir('./csv/*.csv');
    num = size(files);
    numFiles = num(1);
    mdl = 'power_battery_edited';
    for k = 1:length(batterySOCs)
        batterySOC = num2str(batterySOCs(k));
        in = Simulink.SimulationInput(mdl);
        set_param('power_battery_edited/battery','SOC',batterySOC);
        f = waitbar(0,strcat('Computing soc -> ', batterySOC));
        for i = 1:numFiles
            perc = i/numFiles;
            file = files(i).name;
            [filepath,batteryName,ext] = fileparts(file);
            waitbar(perc,f,strcat(batteryName,'@SOC ',batterySOC));
            [ts,stopTime] = load_real_data(batteryName); % ts is required in simulink model
            in = in.setVariable('ts',ts);
            set_param(mdl, 'StopTime',num2str(stopTime) );
            simOut = sim(in);
            generate_synthetic_data( batteryName,batterySOC, simOut.voltage.Data, simOut.SOC.data );
        end
        close(f);
    end
    warning on;
end