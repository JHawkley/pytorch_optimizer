import numpy as np
import pytest
import torch
from torch import nn

from pytorch_optimizer import (
    BSAM,
    GSAM,
    SAM,
    TRAC,
    WSAM,
    CosineScheduler,
    FriendlySAM,
    KohyaHelper,
    Lookahead,
    LookSAM,
    OrthoGrad,
    PCGrad,
    ProportionScheduler,
    ScheduleFreeWrapper,
    load_optimizer,
)
from tests.constants import PULLBACK_MOMENTUM
from tests.utils import (
    Example,
    MultiHeadLogisticRegression,
    Trainer,
    build_model,
    simple_parameter,
    tensor_to_numpy,
)


@pytest.mark.parametrize('pullback_momentum', PULLBACK_MOMENTUM)
def test_lookahead(pullback_momentum, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = Lookahead(load_optimizer('adamw')(model.parameters(), lr=5e-1), pullback_momentum=pullback_momentum)
    optimizer.init_group({})

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=2.0)


@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('wrapper', [SAM, FriendlySAM, LookSAM])
def test_sam_optimizer(adaptive, wrapper, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = wrapper(model.parameters(), load_optimizer('asgd'), lr=5e-1, adaptive=adaptive, use_gc=True)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_sam_style(iterations=3, threshold=2.0)


@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('wrapper', [SAM, FriendlySAM, LookSAM])
def test_sam_optimizer_with_closure(adaptive, wrapper, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = wrapper(model.parameters(), load_optimizer('adamw'), lr=5e-1, adaptive=adaptive)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_with_closure(iterations=3, threshold=2.0)


@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('decouple', [True, False])
def test_wsam_optimizer(adaptive, decouple, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = WSAM(
        model,
        model.parameters(),
        load_optimizer('adamp'),
        lr=5e-2,
        adaptive=adaptive,
        decouple=decouple,
        max_norm=100.0,
    )

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_sam_style(iterations=10, threshold=1.5)


@pytest.mark.parametrize('adaptive', [True, False])
def test_wsam_optimizer_with_closure(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = WSAM(model, model.parameters(), load_optimizer('adamp'), lr=5e-2, adaptive=adaptive, max_norm=100.0)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_wsam_with_closure(iterations=10, threshold=1.5)


@pytest.mark.parametrize('adaptive', [True, False])
def test_gsam_optimizer(adaptive, environment):
    pytest.skip('skip GSAM optimizer')

    x_data, y_data = environment
    model, loss_fn = build_model()

    lr: float = 5e-1
    num_iterations: int = 25

    base_optimizer = load_optimizer('adamp')(model.parameters(), lr=lr)
    lr_scheduler = CosineScheduler(base_optimizer, t_max=num_iterations, max_lr=lr, min_lr=lr, init_lr=lr)
    rho_scheduler = ProportionScheduler(lr_scheduler, max_lr=lr, min_lr=lr)
    optimizer = GSAM(
        model.parameters(), base_optimizer=base_optimizer, model=model, rho_scheduler=rho_scheduler, adaptive=adaptive
    )

    init_loss, loss = np.inf, np.inf
    for _ in range(num_iterations):
        optimizer.set_closure(loss_fn, x_data, y_data)
        _, loss = optimizer.step()

        if init_loss == np.inf:
            init_loss = loss

        lr_scheduler.step()
        optimizer.update_rho_t()

    assert tensor_to_numpy(init_loss) > 1.2 * tensor_to_numpy(loss)


@pytest.mark.parametrize('adaptive', [True, False])
def test_bsam_optimizer(adaptive, environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = BSAM(model.parameters(), lr=2e-3, num_data=len(x_data), rho=1e-5, adaptive=adaptive)

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_with_closure(iterations=20, threshold=1.0)


def test_schedulefree_wrapper():
    model = Example()

    optimizer = ScheduleFreeWrapper(load_optimizer('adamw')(model.parameters(), lr=1e-3, weight_decay=1e-3))
    optimizer.zero_grad()

    model.fc1.weight.grad = torch.randn((1, 1))
    model.norm1.weight.grad = torch.randn((1,))

    with pytest.raises(ValueError):
        optimizer.step()

    optimizer.eval()
    optimizer.train()

    _ = optimizer.__str__
    _ = optimizer.__getstate__()
    _ = optimizer.param_groups

    optimizer.step()

    backup_state = optimizer.state_dict()

    optimizer = ScheduleFreeWrapper(load_optimizer('adamw')(model.parameters(), lr=1e-3, weight_decay=1e-3))
    optimizer.zero_grad()
    optimizer.train()

    optimizer.load_state_dict(backup_state)

    optimizer.step()

    optimizer.eval()
    optimizer.train()
    optimizer.train()

    optimizer.add_param_group({'params': []})


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_pc_grad_optimizers(reduction, environment):
    torch.manual_seed(42)

    x_data, y_data = environment

    model: nn.Module = MultiHeadLogisticRegression()
    loss_fn_1: nn.Module = nn.BCEWithLogitsLoss()
    loss_fn_2: nn.Module = nn.L1Loss()

    optimizer = PCGrad(load_optimizer('adamp')(model.parameters(), lr=1e-1), reduction=reduction)
    optimizer.init_group()

    init_loss, loss = np.inf, np.inf
    for _ in range(5):
        optimizer.zero_grad()

        y_pred_1, y_pred_2 = model(x_data)
        loss1, loss2 = loss_fn_1(y_pred_1, y_data), loss_fn_2(y_pred_2, y_data)

        loss = (loss1 + loss2) / 2.0
        if init_loss == np.inf:
            init_loss = loss

        optimizer.pc_backward([loss1, loss2])
        optimizer.step()

    assert tensor_to_numpy(init_loss) > 1.25 * tensor_to_numpy(loss)


def test_trac_optimizer(environment):
    x_data, y_data = environment
    model, loss_fn = build_model()

    optimizer = TRAC(load_optimizer('adamw')(model.parameters(), lr=1e0))

    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run_trac_style(iterations=3, threshold=2.0)


def test_trac_optimizer_erf_imag():
    model = Example()

    optimizer = TRAC(load_optimizer('adamw')(model.parameters()))
    optimizer.zero_grad()

    complex_tensor = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    optimizer.erf_imag(complex_tensor)

    assert str(optimizer).lower() == 'trac'


@pytest.mark.parametrize('wrapper_optimizer_instance', [Lookahead, OrthoGrad, TRAC])
def test_load_wrapper_optimizer(wrapper_optimizer_instance):
    params = [simple_parameter()]

    _ = wrapper_optimizer_instance(torch.optim.AdamW(params))

    optimizer = wrapper_optimizer_instance(torch.optim.AdamW, params=params)
    optimizer.init_group({'params': []}, updates=[])
    optimizer.zero_grad()

    with pytest.raises(ValueError):
        wrapper_optimizer_instance(torch.optim.AdamW)

    _ = optimizer.param_groups
    _ = optimizer.state

    state = optimizer.state_dict()
    optimizer.load_state_dict(state)


def test_kohya_helper_basic(environment):
    """Test basic KohyaHelper functionality with a simple optimizer."""
    x_data, y_data = environment
    model, loss_fn = build_model()
    
    # Test with raw parameters
    optimizer = KohyaHelper(
        model.parameters(),
        lr=1e-2,
        optimizer_name='adamw',
    )
    
    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=1)


def test_kohya_helper_with_model(environment):
    """Test KohyaHelper when passing a model directly."""
    x_data, y_data = environment
    model, loss_fn = build_model()
    
    # Test with model as first argument (as nn.Module)
    optimizer = KohyaHelper(
        model,
        lr=1e-2,
        optimizer_name='adamw',
    )
    
    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=1)


@pytest.mark.parametrize('use_lookahead', [True, False])
@pytest.mark.parametrize('use_orthograd', [True, False])
def test_kohya_helper_with_wrappers(use_lookahead, use_orthograd, environment):
    """Test KohyaHelper with lookahead and orthograd wrappers."""
    x_data, y_data = environment
    model, loss_fn = build_model()
    
    optimizer = KohyaHelper(
        model.parameters(),
        lr=1e-2,
        optimizer_name='adamw',
        use_lookahead=use_lookahead,
        use_orthograd=use_orthograd,
    )
    
    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=1)


def test_kohya_helper_error_missing_optimizer_name():
    """Test that KohyaHelper raises error when optimizer_name is missing."""
    model = Example()
    
    with pytest.raises(ValueError, match='optimizer_name must be provided'):
        KohyaHelper(model.parameters(), lr=1e-2, optimizer_name=None)


def test_kohya_helper_properties():
    """Test KohyaHelper properties and methods."""
    model = Example()
    
    optimizer = KohyaHelper(
        model.parameters(),
        lr=1e-2,
        optimizer_name='adamw',
    )
    
    # Test properties
    _ = optimizer.param_groups
    _ = optimizer.state
    _ = optimizer.defaults
    
    # Test string representation
    assert str(optimizer) == 'KohyaHelper[AdamW]'
    
    # Test state dict methods
    state = optimizer.state_dict()
    optimizer.load_state_dict(state)
    
    # Test zero_grad
    optimizer.zero_grad()
    
    # Test init_group (should not raise)
    optimizer.init_group({})


def test_kohya_helper_with_parameter_groups():
    """Test KohyaHelper with parameter groups."""
    model = Example()
    
    # Create parameter groups
    param_groups = [
        {'params': model.fc1.parameters(), 'lr': 1e-3},
        {'params': model.norm1.parameters(), 'lr': 1e-4},
    ]
    
    optimizer = KohyaHelper(
        param_groups,
        lr=1e-2,  # lr will be overridden by group lrs
        optimizer_name='adamw',
    )
    
    # Should work without errors
    optimizer.zero_grad()


@pytest.mark.parametrize('special_optimizer', ['adammini', 'muon', 'adamuon', 'adago'])
def test_kohya_helper_model_first_optimizers(special_optimizer, environment):
    """Test KohyaHelper with optimizers that require model as first argument."""
    x_data, y_data = environment
    model, loss_fn = build_model()
    
    # Muon optimizers have special parameter preparation
    # They should work with KohyaHelper's DummyModule
    optimizer = KohyaHelper(
        model.parameters(),
        lr=1e-2,
        optimizer_name=special_optimizer,
    )
    
    # Should create optimizer without errors
    assert optimizer is not None
    
    # Try to run a step
    trainer = Trainer(model, loss_fn, optimizer, x_data, y_data)
    trainer.run(iterations=5, threshold=1)


@pytest.mark.parametrize('lomo_optimizer', ['lomo', 'adalomo'])
def test_kohya_helper_lomo_optimizers(lomo_optimizer, environment):
    """Test KohyaHelper with LOMO family optimizers."""
    model, _ = build_model()
    
    # These optimizers expect model as first argument, not parameters
    # KohyaHelper's DummyModule should handle this
    optimizer = KohyaHelper(
        model.parameters(),
        lr=1e-2,
        optimizer_name=lomo_optimizer,
    )
    
    # Should create optimizer without errors
    assert optimizer is not None
    
    # LOMO optimizers don't implement step() method and require special training workflow
    # We only test that the optimizer can be created, not that it can run training steps
    # This is consistent with how LOMO is tested in test_optimizers.py
