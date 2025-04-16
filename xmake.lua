add_rules("mode.debug", "mode.release")
set_languages("cxx23")
add_requires("opencv", "pugixml")
set_policy("build.c++.gcc.modules.cxx11abi", true)
-- add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

target("main")
    add_files("*.cppm", "*.cc")
    add_packages("opencv", "pugixml")