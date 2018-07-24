function void = generate_synthetic_data( battery,battery_soc, voltage, soc )
%GENERATE_SYNTHETIC_DATA Summary of this function goes here
%   Detailed explanation goes here
    void = -1;
    out = [voltage soc];
    fileForSave = strcat('./synthetic_data/',battery,'_',battery_soc,'.csv');
    csvwrite(fileForSave,out);
    void = 0;
end

