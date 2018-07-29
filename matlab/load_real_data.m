function [ts,stopTime] = load_real_data(path,episodeName )
%LOAD_REAL_DATA Summary of this function goes here
%   Detailed explanation goes here
    loaded = csvread(strcat(path ,'/',episodeName),1,0);
    current = loaded(:,1);
    ts = timeseries(current);
    stopTime = length(current)-1; %simulation starts from 0
end

