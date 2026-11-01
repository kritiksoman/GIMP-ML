on run
    display dialog "Install GIMP-ML for your user account? Fully quit GIMP first (Cmd+Q). Setup requires Python 3.10+ and internet access to download dependencies. No administrator password is needed. This may take a few minutes." buttons {"Cancel", "Install"} default button "Install"
    set gimpApp to "/Applications/GIMP.app"
    try
        POSIX file gimpApp as alias
    on error
        set gimpApp to POSIX path of (choose file with prompt "Select your GIMP 3 application" of type {"com.apple.application-bundle"})
    end try
    set resourceDirectory to POSIX path of (path to resource "install.command")
    set logDirectory to POSIX path of (path to library folder from user domain) & "Logs/GIMP-ML"
    set logFile to logDirectory & "/install.log"
    try
        do shell script "/bin/mkdir -p " & quoted form of logDirectory & " && /bin/chmod 700 " & quoted form of logDirectory & " && /usr/bin/touch " & quoted form of logFile & " && /bin/chmod 600 " & quoted form of logFile & " && /bin/bash " & quoted form of resourceDirectory & " " & quoted form of gimpApp & " > " & quoted form of logFile & " 2>&1"
    on error errorMessage
        display dialog "Installation failed. Check the installation log for details. Make sure GIMP is fully closed and Python 3.10+ is installed." & return & return & errorMessage buttons {"Close", "Show Log"} default button "Show Log" with icon stop
        if button returned of result is "Show Log" then
            do shell script "/usr/bin/open -a TextEdit " & quoted form of logFile
        end if
        return
    end try
    display dialog "GIMP-ML is installed. You can eject this disk image. Open GIMP and choose Filters > GIMP-ML. Use Settings to save your provider API key in Keychain." buttons {"Done"} default button "Done"
end run
