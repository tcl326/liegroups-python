# liegroups-python

`liegroups-python` is a Lie theory library that implements transformations outlined in [A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537).

## Summary
This library is created to aid my understanding of Lie groups through implementation, for a more performant solution, consider using the excellent [manif](https://github.com/artivis/manif) library. This library is also heavily inspired by manif.

This library currently implements the following Lie groups with the following parametrizations

| Group              | Description        | Parametrization |
| ------------------ | ------------------ | --------------- |
| `liegroups.SO2`    | Rotation in 2D     | (theta) angle of rotation |
| `liegroups.SO3`    | Rotation in 3D     | (w1, w2, w3) where norm(**w**) is the angle of rotation around **w** / norm(**w**) axis       |
| `liegroups.SE2`    | Rigid Motion in 2D | (x, y, theta) translation + rotation        |
| `liegroups.SE3`    | Rigid Motion in 3D | (x, y, z, w1, w2, w3) translation + rotation        |


For each group, the following operations are supported

| Operation              | LaTex        | Code |
| ------------------ | ------------------ | --------------- |
| Inverse | ![inverse](https://latex.codecogs.com/svg.latex?\mathbf\mathcal{X}^{-1}) | `X.inverse()` |
| Composition | ![composition](https://latex.codecogs.com/svg.latex?\mathbf\mathcal{X}\circ\mathbf\mathcal{Y}) | `X.compose(Y)` <br> `X @ Y`|
| Group Action | ![act](https://latex.codecogs.com/svg.latex?\mathbf\mathcal{X}\circ\mathbf{v}) | `X.act(v)` <br> `X * v`|
| Exponential Map | ![exp](https://latex.codecogs.com/svg.latex?\exp(\mathbf\varphi^\wedge)) | `G.exp(w)` |
| Logarithm Map | ![log](https://latex.codecogs.com/svg.latex?\log(\mathbf\mathcal{X})^\vee) | `X.log()` |
| Adjoint | ![adj](https://latex.codecogs.com/svg.latex?\operatorname{Adj}(\mathbf\mathcal{X})) | `X.adjoint()` |
| Right plus | ![rplus](https://latex.codecogs.com/svg.latex?\mathbf\mathcal{X}\oplus\mathbf\varphi) | `X.plus(w)` <br> `X + w`|
| Right minus | ![rminus](https://latex.codecogs.com/svg.latex?\mathbf\mathcal{X}\ominus\mathbf\mathcal{Y}) | `X.minus(Y)` <br> `X - Y`
| Left plus | ![lplus](https://latex.codecogs.com/svg.latex?\mathbf\varphi\oplus\mathbf\mathcal{X) | `X.lplus(w)` <br> `w + X`|

- `X`, `Y` represents an instance of the group or a group element.
- `w` represents a group element as Lie algebra expressed in a vector format.
- `G` represents a Lie group.
- `v` represents any vector.

All operations come with their respective analytical Jacobian matrices. To learn more about the Jacobian implemented here, read [this](https://arxiv.org/abs/1812.01537).

## Installation

To install, git clone the repository and run the following commands

```bash
cd liegroups-python
pip install .
```

## Testing

To test the package, ensure that `pytest` is installed. See the [installation guide](https://docs.pytest.org/en/6.2.x/getting-started.html) for more information, and run the following in the repository directory.

```bash
pytest
```