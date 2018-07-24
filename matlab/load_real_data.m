function ts = load_real_data( batteryName )
%LOAD_REAL_DATA Summary of this function goes here
%   Detailed explanation goes here
    loaded = csvread(strcat('./csv/' , batteryName  , '.csv'),1,0);
    current = loaded(:,1);
    ts = timeseries(current);
end

