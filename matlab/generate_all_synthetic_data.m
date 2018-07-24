function [ ] = generate_all_synthetic_data( )
%GENERATE_ALL_SYNTHETIC_DATA Summary of this function goes here
%   Detailed explanation goes here
    warning off;
    files = dir('./csv/*.csv');
    num = size(files);
    numFiles = num(1);
    batterySOC = '80';
    mdl = 'power_battery_edited';
    %open_system(mdl);
    in = Simulink.SimulationInput(mdl);
    f = waitbar(0,strcat('Computing soc -> ', batterySOC));
    for i = 1:numFiles
        perc = i/numFiles;
        file = files(i).name;
        [filepath,batteryName,ext] = fileparts(file);
        waitbar(perc,f,strcat('Computing synthetic data for battery -> ', batteryName));
        ts = load_real_data(batteryName); % ts is required in simulink model
        in = in.setVariable('ts',ts);
        simOut = sim(in);
        generate_synthetic_data( batteryName,batterySOC, simOut.voltage.Data, simOut.SOC.data );
    end
    warning on;
    close(f);
end