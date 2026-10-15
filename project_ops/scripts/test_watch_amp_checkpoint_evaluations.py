from watch_amp_checkpoint_evaluations import planned_iterations


def test_resume_evaluates_each_saved_checkpoint_and_final():
    assert planned_iterations(26998, 29497) == [27000, 27500, 28000, 28500, 29000, 29497]


def test_exact_save_boundary_is_not_duplicated():
    assert planned_iterations(27000, 28000) == [27500, 28000]


def test_long_resume_includes_every_checkpoint_and_non_boundary_final():
    assert planned_iterations(28499, 33498) == list(range(28500, 33001, 500)) + [33498]
