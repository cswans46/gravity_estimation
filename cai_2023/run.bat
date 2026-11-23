@echo off
REM ==========================================================================================
REM  Full pipeline: Cai (2023) gravity on the OECD ICIO, Canadian 4-D split, Toroj SDM.
REM  Put this file in the cai_2023 folder of the repository and double-click it, or run it
REM  from a command prompt:   run.bat            (everything)
REM                           run.bat prepare    (only step 1)   -- also: icio, canada, sdm
REM  Edit the paths below if your files are elsewhere.
REM ==========================================================================================
setlocal EnableExtensions
set "CODE=%~dp0"
set "RAW=D:\Research\Esmaeil Research"
set "IPT_ZIP=%RAW%\12100101-eng.zip"
set "SUT_ZIP=%RAW%\36100478-eng.zip"
set "ODBUS=%CODE%ODBus_v1\ODBus_v1.csv"
REM  provincial boundary file (2021 cartographic, lpr_000b21a_e.zip): r_home and fallback distances
set "BOUNDARIES=%RAW%\lpr_000b21a_e.zip"
REM  2021 Geographic Attribute File (optional but recommended): population-weighted distances
set "GAF=%RAW%\2021_92-151_X.zip"
set "ICIO_DIR=D:\Models\OECD Research"
set "CEPII_DIR=%CODE%"
REM  year of the 4-D split (blank = latest year common to both StatCan tables)
set "YEAR="
REM  MCMC length for the SDM (Toroj uses long chains; reduce for a quick first pass)
set "SDM_ITER=6000"
set "SDM_BURN=3000"
set "SCHULTE_DRAWS=200"
REM ------------------------------------------------------------------------------------------
set "DATA=%CODE%data\canada"
set "SDM=%CODE%data\canada_sdm"
set "RES=%CODE%results"
cd /d "%CODE%"
if not exist "%RES%" mkdir "%RES%"
set "LOG=%RES%\run_log.txt"
echo ===== run started %DATE% %TIME% ===== >> "%LOG%"

where python >nul 2>nul || (echo Python not found on PATH. & goto :fail)
python -c "import gravity_irio" 2>nul || (
  echo Installing the package and its dependencies ...
  python -m pip install -e . || goto :fail
)

set "STEP=%~1"
if "%STEP%"=="" set "STEP=all"

REM ---------------------------------------------------------------- 1. StatCan data
if /i "%STEP%"=="all" goto :prepare
if /i "%STEP%"=="prepare" goto :prepare
goto :after_prepare
:prepare
for %%F in ("%IPT_ZIP%" "%SUT_ZIP%") do if not exist "%%~F" (echo Missing %%~F & goto :fail)
set "GAFARG="
if exist "%GAF%" (set "GAFARG=--gaf "%GAF%"") else (echo NOTE: GAF not found - using the capitals/land-area fallback distances.)
set "BNDARG="
if exist "%BOUNDARIES%" (set "BNDARG=--boundaries "%BOUNDARIES%"") else (echo NOTE: boundary zip not found at %BOUNDARIES%.)
set "YEARARG="
if not "%YEAR%"=="" set "YEARARG=--year %YEAR%"
echo [1] Converting StatCan tables ...
python prepare_data.py --ipt-zip "%IPT_ZIP%" --sut-zip "%SUT_ZIP%" %BNDARG% %GAFARG% %YEARARG% --out "%DATA%" --sdm-out "%SDM%" 2>&1
if errorlevel 3 if not errorlevel 4 (
  echo.
  echo STOP: edit the 'role' column of %DATA%\sut_member_roles.csv for the members listed above,
  echo       then run run.bat again. If the layout looks wrong, send %DATA%\sut_describe.txt.
  goto :end
)
if errorlevel 1 goto :fail
if /i "%STEP%"=="prepare" goto :end
:after_prepare

REM ---------------------------------------------------------------- 2. ICIO elasticities (Cai 2023)
if /i "%STEP%"=="all" goto :icio
if /i "%STEP%"=="icio" goto :icio
goto :after_icio
:icio
echo [2] Cai (2023) eq. 9 on the OECD ICIO 2025 panel ...
python estimation.py --data-dir "%ICIO_DIR%" --cepii-dir "%CEPII_DIR%" --level both --out "%RES%" || goto :fail
if /i "%STEP%"=="icio" goto :end
:after_icio

REM ---------------------------------------------------------------- 3. Canadian elasticities and 4-D split
if /i "%STEP%"=="all" goto :canada
if /i "%STEP%"=="canada" goto :canada
goto :after_canada
:canada
echo [3a] Interprovincial elasticities (population distances) ...
python canada_elasticities.py --canada-dir "%DATA%" --slope-by none --out "%RES%" || goto :fail
python canada_elasticities.py --canada-dir "%DATA%" --slope-by origin --out "%RES%" || goto :fail
if exist "%ODBUS%" (
  echo [3b] Sensitivity: ODBus activity-weighted distances where ODBus has coverage ...
  python canada_elasticities.py --canada-dir "%DATA%" --distances odbus --odbus "%ODBUS%" --slope-by none --out "%RES%" || goto :fail
) else (echo NOTE: ODBus not found at %ODBUS% - skipping the ODBus sensitivity.)
echo [3c] Concordances SUT to ICIO 2025 and NAICS 2-digit ...
python make_concordances.py --canada-dir "%DATA%" || goto :fail
echo [3d] 4-D split Z[i,j,b,u] ...
if exist "%RES%\cai_pair.csv" (
  python canada_disaggregation.py --canada-dir "%DATA%" --results "%RES%" --out "%RES%\split_icio_pair" --delta-source icio-pair --schulte-draws %SCHULTE_DRAWS% || goto :fail
  python canada_disaggregation.py --canada-dir "%DATA%" --results "%RES%" --out "%RES%\split_icio_sector" --delta-source icio-sector || goto :fail
) else (echo NOTE: results\cai_pair.csv missing - run step "icio" first for the ICIO-elasticity splits.)
python canada_disaggregation.py --canada-dir "%DATA%" --results "%RES%" --out "%RES%\split_ipt" --delta-source ipt --ipt-slope none || goto :fail
python canada_disaggregation.py --canada-dir "%DATA%" --results "%RES%" --out "%RES%\split_ipt_origin" --delta-source ipt --ipt-slope origin || goto :fail
if exist "%RES%\canada_sector_pair_distances.npy" (
  python canada_disaggregation.py --canada-dir "%DATA%" --results "%RES%" --out "%RES%\split_ipt_odbus" --delta-source ipt --ipt-slope none_odbus --distances user-specific || goto :fail
)
python canada_disaggregation.py --canada-dir "%DATA%" --results "%RES%" --out "%RES%\split_provincial_tables" --delta-source ipt --trade-source provincial-tables || goto :fail
if /i "%STEP%"=="canada" goto :end
:after_canada

REM ---------------------------------------------------------------- 4. Spatial Durbin IRIO (Toroj)
if /i "%STEP%"=="all" goto :sdm
if /i "%STEP%"=="sdm" goto :sdm
goto :end
:sdm
echo [4] Toroj Bayesian spatial Durbin IRIO with sensitivities ...
set "BNDARG="
if exist "%BOUNDARIES%" set "BNDARG=--boundaries "%BOUNDARIES%""
python spatial_irio.py --sdm-dir "%SDM%" --prior toroj %BNDARG% --iter %SDM_ITER% --burn %SDM_BURN% --sensitivity --out "%RES%\sdm" || goto :fail

:end
echo ===== finished %DATE% %TIME% ===== >> "%LOG%"
echo Done. Results are in %RES%
endlocal
exit /b 0
:fail
echo.
echo FAILED - see the messages above. Send them (and %DATA%\sut_describe.txt if step 1 failed).
endlocal
exit /b 1
