import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_dmg, install


class InstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = self.root / "GIMP/3.2"
        self.config.mkdir(parents=True)
        self.directory = self.config / "plug-ins"
        self.gimprc = self.config / "gimprc"

    def test_discovery_uses_gimp_version_directory_and_search_path(self):
        output = "GIMP startup message\nGIMPML_CONFIG=" + json.dumps(
            [str(self.config), str(self.directory)]) + "\n"
        with patch("scripts.install.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, stdout=output)) as run:
            self.assertEqual(install.gimp_configuration("/custom/GIMP"),
                             (self.config, str(self.directory)))
        self.assertEqual(run.call_args.args[0][0], "/custom/GIMP")
        self.assertTrue(run.call_args.kwargs["check"])

    def test_discovery_fails_without_marker(self):
        with patch("scripts.install.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, stdout="no configuration")), self.assertRaises(RuntimeError):
            install.gimp_configuration("/custom/GIMP")

    def test_existing_search_path_does_not_modify_preferences(self):
        original = '# comment\n(plug-in-path "${gimp_dir}/plug-ins:other")\n'
        self.gimprc.write_text(original)
        install.register_plugin_path(self.config, self.directory,
                                     str(self.directory) + os.pathsep + "other")
        self.assertEqual(self.gimprc.read_text(), original)
        self.assertFalse(self.gimprc.with_name("gimprc.gimpml-backup").exists())

    def test_adds_path_and_preserves_variables_and_other_preferences(self):
        original = '# comment\n(plug-in-path "${gimp_plug_in_dir}/plug-ins:custom")\n(theme "Dark")\n'
        self.gimprc.write_text(original)
        install.register_plugin_path(self.config, self.directory, "system:custom")
        updated = self.gimprc.read_text()
        self.assertIn('${gimp_plug_in_dir}/plug-ins:custom:' + str(self.directory), updated)
        self.assertIn('(theme "Dark")', updated)
        self.assertEqual(self.gimprc.with_name("gimprc.gimpml-backup").read_text(), original)
        install.register_plugin_path(self.config, self.directory,
                                     "system:custom:" + str(self.directory))
        self.assertEqual(self.gimprc.read_text(), updated)

    def test_missing_property_preserves_effective_defaults(self):
        self.gimprc.write_text('(theme "Dark")\n')
        install.register_plugin_path(self.config, self.directory, "user:system")
        self.assertIn('"user:system:' + str(self.directory) + '"', self.gimprc.read_text())

    def test_missing_config_is_created_without_backup(self):
        install.register_plugin_path(self.config, self.directory, "system")
        self.assertTrue(self.gimprc.is_file())
        self.assertFalse(self.gimprc.with_name("gimprc.gimpml-backup").exists())

    def test_quoted_directory_is_escaped(self):
        directory = self.root / 'a "quoted" folder'
        install.register_plugin_path(self.config, directory, "system")
        self.assertIn(r'a \"quoted\" folder', self.gimprc.read_text())

    def test_unsafe_preferences_are_not_modified(self):
        for original in ('(plug-in-path invalid)\n',
                         '(plug-in-path "a")\n(plug-in-path "b")\n'):
            with self.subTest(original=original):
                self.gimprc.write_text(original)
                with self.assertRaises(RuntimeError):
                    install.register_plugin_path(self.config, self.directory, "other")
                self.assertEqual(self.gimprc.read_text(), original)

    def test_path_separator_in_directory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            install.register_plugin_path(self.config, self.root / ("a" + os.pathsep + "b"), "other")
        self.assertFalse(self.gimprc.exists())

    def test_running_gimp_and_process_check_errors_are_reported(self):
        for code in (0, 2):
            with self.subTest(code=code), patch(
                    "scripts.install.subprocess.run",
                    return_value=subprocess.CompletedProcess([], code, stderr="error")):
                with self.assertRaises(RuntimeError):
                    install.require_gimp_closed()
        with patch("scripts.install.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 1, stderr="")):
            install.require_gimp_closed()

    def test_permanent_runtime_survives_source_removal(self):
        source = self.root / "mounted-payload"
        (source / "gimpml").mkdir(parents=True)
        (source / "gimpml/__init__.py").write_text("")
        (source / "gimpml/requirements.txt").write_text("requests\n")
        plugin_source = source / "plug-ins/gimpml-gemini/gimpml-gemini.py"
        plugin_source.parent.mkdir(parents=True)
        plugin_source.write_text("#!/usr/bin/env python3\n")
        (source / ".env").write_text("DO_NOT_COPY=secret\n")
        runtime = self.root / "permanent backend"
        python = runtime / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        def create_environment(_environment):
            python.parent.mkdir(parents=True)
            python.write_text("")

        with (patch("scripts.install.ROOT", source),
              patch("sys.argv", ["install.py", "--plug-in-dir", str(self.directory),
                                 "--runtime-dir", str(runtime)]),
              patch("scripts.install.venv.EnvBuilder.create", side_effect=create_environment),
              patch("scripts.install.subprocess.run")):
            install.main()
        source.rename(self.root / "ejected-payload")
        config = json.loads((self.directory / "gimpml-gemini/runtime.json").read_text())
        self.assertEqual(config, {"root": str(runtime), "python": str(python)})
        self.assertTrue(Path(config["python"]).is_file())
        self.assertTrue((Path(config["root"]) / "gimpml/__init__.py").is_file())
        self.assertFalse((runtime / ".env").exists())
        target = self.directory / "gimpml-gemini/gimpml-gemini.py"
        self.assertEqual(target.stat().st_mode & 0o777, 0o755)

    def test_runtime_inside_source_package_is_rejected(self):
        with patch("scripts.install.ROOT", self.root), self.assertRaisesRegex(
                RuntimeError, "inside the source"):
            install.copy_runtime(self.root / "gimpml/backend")
        self.assertFalse((self.root / "gimpml").exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS preference registration")
    def test_main_wires_registration_for_custom_plugin_directory(self):
        directory = self.root / "custom plug-ins"
        python = install.ROOT / ".venv/bin/python"
        with (patch("sys.argv", ["install.py", "--plug-in-dir", str(directory),
                                 "--register-plugin-path"]),
              patch("scripts.install.gimp_configuration", return_value=(self.config, "system")),
              patch("scripts.install.require_gimp_closed") as closed,
              patch("scripts.install.register_plugin_path") as register,
              patch("scripts.install.subprocess.run"),
              patch("scripts.install.venv.EnvBuilder.create"),
              patch("scripts.install.Path.is_file", return_value=True)):
            install.main()
        self.assertEqual(closed.call_count, 2)
        register.assert_called_once_with(self.config, directory, "system")
        config = json.loads((directory / "gimpml-gemini/runtime.json").read_text())
        self.assertEqual(config["python"], str(python))

    @unittest.skipUnless(sys.platform == "darwin", "macOS process detection")
    def test_busy_gimp_stops_before_copying_backend(self):
        runtime = self.root / "backend"
        with (patch("sys.argv", ["install.py", "--runtime-dir", str(runtime),
                                 "--register-plugin-path"]),
              patch("scripts.install.require_gimp_closed", side_effect=RuntimeError("Quit GIMP")),
              self.assertRaisesRegex(RuntimeError, "Quit GIMP")):
            install.main()
        self.assertFalse(runtime.exists())

    def test_dmg_staging_excludes_environment_and_seals_resources(self):
        def run(command, **_kwargs):
            if command[0] == "osacompile":
                (Path(command[2]) / "Contents/Resources").mkdir(parents=True)

        with patch("scripts.build_dmg.subprocess.run", side_effect=run) as command:
            application = build_dmg.stage_installer(self.root)
        payload = application / "Contents/Resources/payload"
        self.assertEqual({path.name for path in payload.iterdir()}, {"gimpml", "plug-ins", "scripts"})
        self.assertFalse(list(payload.rglob("__pycache__")))
        self.assertFalse(list(payload.rglob("runtime.json")))
        self.assertTrue((payload / "scripts/install.py").is_file())
        self.assertEqual(command.call_args_list[-2].args[0][:4],
                         ["codesign", "--force", "--sign", "-"])
        self.assertEqual(command.call_args_list[-1].args[0][:4],
                         ["codesign", "--verify", "--deep", "--strict"])


if __name__ == "__main__":
    unittest.main()
