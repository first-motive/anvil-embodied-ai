"""The VLA sim launches, loaded from the installed share and set up as fm-sim would.

fm-sim's run.launch.py includes ``sim_<policy>.launch.py`` with ``use_sim``,
``robot_profile`` and ``scenario``; each wrapper includes ``vla_sim.launch.py`` with
its ``policy``.
"""

import importlib.util
import json
import os

import pytest
import yaml

launch = pytest.importorskip("launch")
pytest.importorskip("launch_ros.actions")
packages = pytest.importorskip("ament_index_python.packages")

PROFILE = "fm_openarm_v2"
POLICIES = ("act", "smolvla", "pi0")


def _load(file_name):
    try:
        share = packages.get_package_share_directory("lerobot_control")
    except packages.PackageNotFoundError:
        pytest.skip("lerobot_control is not installed; run under colcon test")
    path = os.path.join(share, "launch", file_name)
    spec = importlib.util.spec_from_file_location(file_name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vla_sim():
    return _load("vla_sim.launch.py")


def _checkpoint(root, policy, declared_type=None, nested=False, inference=True):
    checkpoint = root / policy
    model_dir = checkpoint / "pretrained_model" if nested else checkpoint
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"type": declared_type or policy}))
    if inference:
        (checkpoint / "inference.yaml").write_text("joint_state_topic: /joint_states\n")
    return checkpoint


def _declared_in(description):
    return {
        action.name: action
        for action in description.entities
        if isinstance(action, launch.actions.DeclareLaunchArgument)
    }


def _declared(module):
    return _declared_in(module.generate_launch_description())


def _setup(vla_sim, **arguments):
    context = launch.LaunchContext()
    for name, declared in _declared(vla_sim).items():
        if declared.default_value is not None:
            context.launch_configurations[name] = declared.default_value[0].text
    context.launch_configurations.update(arguments)
    return context, vla_sim._launch_setup(context)


def _parameters(node, context):
    # launch_ros has no public getter for a node's parameters; see fm-teleop's twin.
    node._perform_substitutions(context)
    expanded = getattr(node, "_Node__expanded_parameter_arguments", None)
    assert expanded is not None, "launch_ros no longer has Node.__expanded_parameter_arguments"
    parameters = {}
    for path, is_file in expanded:
        assert is_file
        with open(path) as handle:
            parameters.update(yaml.safe_load(handle)["/**"]["ros__parameters"])
    return parameters


@pytest.mark.parametrize("policy", POLICIES)
def test_wrapper_forwards_the_run_arguments_to_vla_sim(policy, tmp_path, monkeypatch):
    # Visit the wrapper as fm-sim's include would: its declared arguments set from the
    # run, then the include itself, then vla_sim's setup. The include refuses to load a
    # file whose required arguments it does not name, so this fails if one is dropped.
    _checkpoint(tmp_path, policy)
    monkeypatch.setenv("LEROBOT_SIM_CHECKPOINTS", str(tmp_path))
    wrapper = _load(f"sim_{policy}.launch.py").generate_launch_description()
    *declares, include = wrapper.entities
    assert {action.name for action in declares} == {"use_sim", "robot_profile", "scenario"}
    assert isinstance(include, launch.actions.IncludeLaunchDescription)

    context = launch.LaunchContext()
    run = {"use_sim": "true", "robot_profile": PROFILE, "scenario": "empty"}
    context.launch_configurations.update(run)
    # visit returns the actions that set the forwarded arguments, then the description.
    *set_arguments, included = include.visit(context)
    for action in set_arguments:
        action.visit(context)
    assert isinstance(included, launch.LaunchDescription)
    assert include.launch_description_source.location == os.path.join(
        packages.get_package_share_directory("lerobot_control"), "launch", "vla_sim.launch.py"
    )
    assert context.launch_configurations == {**run, "policy": policy}
    setup = [e for e in included.entities if isinstance(e, launch.actions.OpaqueFunction)]
    (node,) = setup[0].execute(context)
    assert node.node_executable == "inference_node"


@pytest.mark.parametrize("policy", POLICIES)
def test_sim_run_starts_inference_node_on_the_policys_checkpoint(
    vla_sim, policy, tmp_path, monkeypatch
):
    checkpoint = _checkpoint(tmp_path, policy, nested=policy == "pi0")
    monkeypatch.setenv(vla_sim.CHECKPOINTS_ENV, str(tmp_path))
    context, (node,) = _setup(vla_sim, policy=policy, use_sim="true", robot_profile=PROFILE)
    assert (node.node_package, node.node_executable) == ("lerobot_control", "inference_node")
    assert _parameters(node, context) == {
        "model_path": str(checkpoint.resolve()),
        "config_file": str((checkpoint / "inference.yaml").resolve()),
        "robot_profile": PROFILE,
        "use_sim_time": True,
    }


@pytest.mark.parametrize("arguments", [{}, {"use_sim": "false"}])
def test_refuses_to_start_outside_the_sim(vla_sim, tmp_path, monkeypatch, arguments):
    # A wrapper run by hand defaults to use_sim:=false; it must never reach hardware.
    _checkpoint(tmp_path, "act")
    monkeypatch.setenv(vla_sim.CHECKPOINTS_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="use_sim must be 'true'$"):
        _setup(vla_sim, policy="act", robot_profile=PROFILE, **arguments)


def test_refuses_without_a_checkpoint_root(vla_sim, monkeypatch):
    monkeypatch.delenv(vla_sim.CHECKPOINTS_ENV, raising=False)
    with pytest.raises(ValueError, match="no sim checkpoint: set LEROBOT_SIM_CHECKPOINTS"):
        _setup(vla_sim, policy="act", use_sim="true", robot_profile=PROFILE)


@pytest.mark.parametrize("policy", ["", "diffusion", "../act", "ACT"])
def test_refuses_an_unknown_policy(vla_sim, policy, tmp_path):
    with pytest.raises(ValueError, match="policy must be one of: act, smolvla, pi0$"):
        vla_sim.resolve_checkpoint(policy, str(tmp_path))


def test_refuses_a_missing_checkpoint(vla_sim, tmp_path):
    with pytest.raises(FileNotFoundError, match="no inference.yaml"):
        vla_sim.resolve_checkpoint("act", str(tmp_path))


def test_refuses_a_checkpoint_without_its_inference_config(vla_sim, tmp_path):
    _checkpoint(tmp_path, "act", inference=False)
    with pytest.raises(FileNotFoundError, match="no inference.yaml"):
        vla_sim.resolve_checkpoint("act", str(tmp_path))


def test_refuses_a_checkpoint_without_config_json(vla_sim, tmp_path):
    (tmp_path / "act").mkdir()
    (tmp_path / "act" / "inference.yaml").write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="no config.json"):
        vla_sim.resolve_checkpoint("act", str(tmp_path))


def test_refuses_a_checkpoint_of_another_policy(vla_sim, tmp_path):
    _checkpoint(tmp_path, "smolvla", declared_type="act")
    with pytest.raises(ValueError, match="config.json declares type 'act', expected 'smolvla'"):
        vla_sim.resolve_checkpoint("smolvla", str(tmp_path))


@pytest.mark.parametrize("value", ["", "True", "yes", "1"])
def test_refuses_a_use_sim_other_than_true_or_false(vla_sim, value):
    with pytest.raises(ValueError, match="use_sim must be one of: true, false$"):
        _setup(vla_sim, policy="act", use_sim=value, robot_profile=PROFILE)


@pytest.mark.parametrize("argument", ["robot_profile", "scenario"])
@pytest.mark.parametrize("value", ["", "../fm_openarm_v2", "a/b", "9lives", "x y"])
def test_refuses_a_malformed_name_without_echoing_it(vla_sim, argument, value):
    arguments = {"policy": "act", "use_sim": "true", "robot_profile": PROFILE, argument: value}
    with pytest.raises(ValueError, match=f"^{argument} must be a name") as raised:
        _setup(vla_sim, **arguments)
    assert repr(value) not in str(raised.value)


def test_refuses_an_unknown_robot_profile(vla_sim):
    with pytest.raises(FileNotFoundError, match="robot profile 'no_such_robot' not found"):
        _setup(vla_sim, policy="act", use_sim="true", robot_profile="no_such_robot")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_refuses_an_unreadable_config_json(vla_sim, tmp_path, content):
    checkpoint = _checkpoint(tmp_path, "act")
    (checkpoint / "config.json").write_text(content)
    with pytest.raises(ValueError, match="config.json"):
        vla_sim.resolve_checkpoint("act", str(tmp_path))


@pytest.mark.parametrize("file_name", ["inference.yaml", "config.json"])
def test_refuses_a_checkpoint_file_symlinked_outside_it(vla_sim, tmp_path, file_name):
    checkpoint = _checkpoint(tmp_path / "root", "act")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / file_name
    target.write_bytes((checkpoint / file_name).read_bytes())
    (checkpoint / file_name).unlink()
    (checkpoint / file_name).symlink_to(target)
    with pytest.raises(PermissionError, match=f"{file_name} in sim checkpoint .* resolves outside"):
        vla_sim.resolve_checkpoint("act", str(tmp_path / "root"))


def test_accepts_a_policy_dir_that_is_itself_a_symlink(vla_sim, tmp_path):
    # The checkpoint dir is resolved first, so linking the whole dir in is allowed.
    real = _checkpoint(tmp_path / "store", "act")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "act").symlink_to(real)
    model_path, _ = vla_sim.resolve_checkpoint("act", str(tmp_path / "root"))
    assert model_path == str(real.resolve())


def test_refuses_an_oversized_config_json(vla_sim, tmp_path):
    checkpoint = _checkpoint(tmp_path, "act")
    padding = " " * vla_sim.MAX_MODEL_CONFIG_BYTES
    (checkpoint / "config.json").write_text('{"type": "act"}' + padding)
    with pytest.raises(ValueError, match="config.json is over 1048576 bytes"):
        vla_sim.resolve_checkpoint("act", str(tmp_path))
