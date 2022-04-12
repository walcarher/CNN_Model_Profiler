# Variable names
# To run this script use:
# quartus_sh --script measurements_config.tcl
set project_name "AlgoBlocsMeasurements"
set delirium_dist_lib_hdl_dirpath "../delirium_dist/lib/hdl/"
set vhdl_generated_dirpath "vhdl_generated/"
set sdc_file_dirpath "measurements_config.sdc"

# Create project if not existent or open it if it does exist 
if {[project_exists $project_name]} {
	project_open $project_name
} else {
	project_new $project_name
}

# INITIALIZATION
# Set global assignments for project configuration and synthesis
set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL
set_global_assignment -name FAMILY "Cyclone 10 GX"
set_global_assignment -name DEVICE 10CX220YF780E5G
set_global_assignment -name ORIGINAL_QUARTUS_VERSION 17.1.0
set_global_assignment -name LAST_QUARTUS_VERSION "17.1.0 Pro Edition"
# Toolchain and properties for the DevKit (probably needs to be replaced later on):
set_global_assignment -name MIN_CORE_JUNCTION_TEMP 0
set_global_assignment -name MAX_CORE_JUNCTION_TEMP 100
set_global_assignment -name POWER_AUTO_COMPUTE_TJ ON
set_global_assignment -name POWER_PRESET_COOLING_SOLUTION "23 MM HEAT SINK WITH 200 LFPM AIRFLOW"
set_global_assignment -name POWER_BOARD_THERMAL_MODEL "NONE (CONSERVATIVE)"
# Enable Power Analyzer Tool for power dissipation estimation
set_global_assignment -name POWER_USE_PVA ON
set_global_assignment -name FLOW_ENABLE_POWER_ANALYZER ON
set_global_assignment -name POWER_DEFAULT_INPUT_IO_TOGGLE_RATE "90 %"
set_global_assignment -name POWER_USE_INPUT_FILES OFF

# Set Timing Analyzer Tool for latency estimation
set_global_assignment -name SDC_FILE $sdc_file_dirpath
# Set compiler version to VHDL 2008 as required by Delirium
set_global_assignment -name VHDL_INPUT_VERSION VHDL_2008
# Add all .vhd files from delirium library on delirium_dist/lib/hdl
set all_vhdl_delirium_lib_files [glob -directory $delirium_dist_lib_hdl_dirpath -- "*.vhd"]
foreach vhdl_file $all_vhdl_delirium_lib_files {
	set_global_assignment -name VHDL_FILE $vhdl_file
}

# ITERATIONS OVER CONVOLUTIONS, IMAGE SIZE, CHANNELS AND CASES START HERE
# Iterate over all convkxk  directories
set convsDir [glob -nocomplain -type {d  r} -path $vhdl_generated_dirpath *] 
foreach convDir $convsDir {
	# Iterate over every hw directory of each convkxk directory 
	set hwsDir [glob -type d -directory $convDir *]
	foreach hwDir $hwsDir {
		# Iterate for over every c directory of each hw image size
		set csDir [glob -type d -directory $hwDir *]
		foreach cDir $csDir {
			# Iterate over every case n of each channel depth c
			set casesDir [glob -type d -directory $cDir *]
			foreach caseDir $casesDir {
				# Add all .vhd files from generated VHDL files from the ONNX models
				# This adds the top level vhdl main file, parameter file and bitwidth file
				set all_vhdl_generated_files [glob -directory $caseDir -- "*.vhd"]
				foreach vhdl_file $all_vhdl_generated_files {
					set_global_assignment -name VHDL_FILE $vhdl_file
				}
				# Set main generated vhdl file from delirium as top level entity 
				set top_level_main_file [glob -directory $caseDir "*main.vhd"]
				# Remove prefix from top level file string
				set top_level_main_file [string replace $top_level_main_file 0 [string length $caseDir]-0]
				# Remove suffix from top level file string
				set top_level_main_file [string replace $top_level_main_file end-3 end ""]
				puts $top_level_main_file
				set_global_assignment -name TOP_LEVEL_ENTITY $top_level_main_file
		
				# Load flow package, compile all project and generate .sof programming file
				load_package flow
				if {[catch {execute_flow -compile} errmsg]} {
					puts "Error: $top_level_main_file couldn't be compiled\n"
					puts "Error Info: $errmsg\s"
				} else {
					# Report compilation results
					load_package report
					load_report
					# Save compilation summary to resources.csv
					set panel_name "Fitter||Place Stage||Fitter Resource Usage Summary"
					set csv_file "${caseDir}_resources.csv"
					set fh [open $csv_file w]
					set num_rows [get_number_of_rows -name $panel_name]
					for {set i 0} {$i < $num_rows} {incr i} {
						set row_data [get_report_panel_row -name $panel_name -row $i]
						puts $fh [join [lmap elem $row_data {format {"%s"} [string map {{"} {""}} $elem]}] ,]
					}
					close $fh
					# Save power compilation summary to *power.csv
					set panel_name "Power Analyzer||Power Analyzer Summary"
					set csv_file "${caseDir}_power.csv"
					set fh [open $csv_file w]
					set num_rows [get_number_of_rows -name $panel_name]
					for {set i 0} {$i < $num_rows} {incr i} {
						set row_data [get_report_panel_row -name $panel_name -row $i]
						puts $fh [join [lmap elem $row_data {format {"%s"} [string map {{"} {""}} $elem]}] ,]
					}
					close $fh
					# Save max frequency summary to *maxfreq.csv
					set panel_name "TimeQuest Timing Analyzer||Slow 900mV 100C Model||Slow 900mV 100C Model Fmax Summary"
					set csv_file "${caseDir}_fmax.csv"
					set fh [open $csv_file w]
					set num_rows [get_number_of_rows -name $panel_name]
					for {set i 0} {$i < $num_rows} {incr i} {
						set row_data [get_report_panel_row -name $panel_name -row $i]
						puts $fh [join [lmap elem $row_data {format {"%s"} [string map {{"} {""}} $elem]}] ,]
					}
					close $fh
					unload_report
					# Copy .sof file to case folder
					file copy -force "${project_name}.sof" "${caseDir}.sof"
				}
				# Remove generated vhdl files for next case/layer iteration
				foreach vhdl_file $all_vhdl_generated_files {
					set_global_assignment -remove -name VHDL_FILE $vhdl_file
				}
			}
		}
	}
}

# ITERATION ENDS HERE

project_close
