# Furnace cavity
Package-style documentation for an effusive atomic beam through a rectangular cavity (molecular-flow Monte Carlo)

Author: Kelvin Titimbo 
Language: Julia  

---

## Contents
1. [Overview](#overview)  
2. [Geometry and Coordinate System](#geometry-and-coordinate-system)  
3. [Mathematical Model](#mathematical-model)  
4. [API Reference](#api-reference)  
   - [Effusive Source Sampling](#effusive-source-sampling)  
   - [Geometry Helpers](#geometry-helpers)  
   - [Lambertian Scattering](#lambertian-scattering)  
   - [Transport Models](#transport-models)  
   - [Analysis Utilities](#analysis-utilities)  
5. [Typical Workflow](#typical-workflow)  
6. [Numerical Notes](#numerical-notes)  

---

# Overview

This package provides a minimal set of building blocks to simulate an effusive atomic beam entering a **rectangular cavity** and propagating to an exit plane under molecular-flow assumptions.

Core features:
- Effusive **initial velocity** sampling and uniform **initial position** sampling on the entrance slit.
- Exact **ballistic propagation** between boundaries using ray–plane intersections.
- **Diffuse (Lambertian)** scattering on side walls (cosine-weighted hemisphere).
- Two transport models:
  - **Diffusive model**: Lambertian scattering on x/z walls
  - **Ballistic model**: x/z walls are absorbing (geometric acceptance baseline)
- Post-processing:
  - exit angular distributions,
  - transmission probabilities,
  - Clausing factor (as used in this workflow),
  - brightness proxy within a forward acceptance cone.

---

# Geometry and Coordinate System

The cavity is aligned with the transport axis \(y\):

- Entrance plane: \(y = 0\)  
- Exit plane: \(y = L\)

The slit is centered at \((x,z)=(0,0)\) and spans:

\[
x \in \left[-\frac{x_\mathrm{furnace}}{2}, +\frac{x_\mathrm{furnace}}{2}\right],\qquad
z \in \left[-\frac{z_\mathrm{furnace}}{2}, +\frac{z_\mathrm{furnace}}{2}\right].
\]

Slit area (consistent with the above bounds):

\[
A = x_\mathrm{furnace}\,z_\mathrm{furnace}.
\]

---

# Mathematical Model

## Free flight
Between wall interactions, each particle follows:

\[
\mathbf{r}(t) = \mathbf{r}_0 + \mathbf{v}\,t,
\]

with constant velocity \(\mathbf{v}\).

## Plane intersection
To find when a particle reaches a coordinate plane \(r_k = b\) (with \(k \in \{x,y,z\}\)), solve:

\[
r_{0,k} + v_k t = b
\quad\Rightarrow\quad
t = \frac{b - r_{0,k}}{v_k}.
\]

Only **future** intersections are physically relevant: \(t > 0\).

## Lambertian (diffuse) wall scattering
When a particle hits a side wall, its direction is resampled from a cosine-weighted distribution over the inward hemisphere:

\[
p(\theta)\,\mathrm{d}\Omega \propto \cos\theta\,\mathrm{d}\Omega,
\qquad 0 \le \theta \le \frac{\pi}{2},
\]

where \(\theta\) is the angle from the inward surface normal.

The speed \(|\mathbf{v}|\) is preserved (elastic reflection), only the direction changes.

---

# API Reference

## Effusive Source Sampling

### `AtomicBeamVelocity_v3(rng, p) -> SVector{3,Float64}`

```julia
@inline function AtomicBeamVelocity_v3(rng::AbstractRNG, p::EffusionParams)::SVector{3,Float64}
    ϕ = TWOπ * rand(rng)
    θ = asin(sqrt(rand(rng)))
    v = sqrt(-2*p.α2 * (1 + lambertw((rand(rng)-1)*INV_E, -1)))
    sθ = sin(θ); cθ = cos(θ); sϕ = sin(ϕ); cϕ = cos(ϕ)
    return SVector(v*sθ*sϕ, v*cθ, v*sθ*cϕ)
end
```

**Purpose**  
Samples an **effusive** velocity vector \(\mathbf{v}=(v_x,v_y,v_z)\) leaving the source into the forward half-space.

**Angular sampling**  
Azimuth:
\[
\phi \sim \mathrm{Uniform}(0,2\pi).
\]

Polar angle (cosine-weighted about \(+y\)):
\[
\theta = \arcsin\!\left(\sqrt{u}\right),\qquad u\sim \mathrm{Uniform}(0,1).
\]

This yields the Lambertian angular distribution relative to the emission normal (here aligned with \(+y\)).

**Speed sampling**  
A speed \(v\) is sampled by an inverse transform involving the Lambert W function (branch \(-1\)):

\[
v = \sqrt{-2\alpha^2\left(1 + W_{-1}\!\left(\frac{u-1}{e}\right)\right)},
\qquad u\sim \mathrm{Uniform}(0,1).
\]

Here \(\alpha^2 = p.\alpha2\) sets the velocity scale.

**Return value**  
Velocity components (with \(+y\) forward):
\[
v_x = v\sin\theta\,\sin\phi,\quad
v_y = v\cos\theta,\quad
v_z = v\sin\theta\,\cos\phi.
\]

---

### `InitialPositions(rng) -> SVector{3,Float64}`

```julia
@inline function InitialPositions(rng::AbstractRNG)
    x0 = x_furnace * (rand(rng) - 0.5)
    z0 = z_furnace * (rand(rng) - 0.5)
    return SVector(x0,0,z0)
end
```

**Purpose**  
Samples a uniform position on the entrance slit (plane \(y=0\)).

**Distribution**
\[
x_0 \sim \mathrm{Uniform}\!\left(-\frac{x_\mathrm{furnace}}{2}, +\frac{x_\mathrm{furnace}}{2}\right),\qquad
z_0 \sim \mathrm{Uniform}\!\left(-\frac{z_\mathrm{furnace}}{2}, +\frac{z_\mathrm{furnace}}{2}\right),\qquad
y_0 = 0.
\]

---

### `sample_initial_conditions(N, rng, p) -> (pos, vel)`

```julia
@inline function sample_initial_conditions(N, rng, p)
    pos = Matrix{Float64}(undef, N, 3)
    vel = Matrix{Float64}(undef, N, 3)

    @inbounds for i in 1:N
        pos_i = InitialPositions(rng)
        vel_i = AtomicBeamVelocity_v3(rng, p)

        pos[i,1] = pos_i[1]; pos[i,2] = pos_i[2]; pos[i,3] = pos_i[3]
        vel[i,1] = vel_i[1]; vel[i,2] = vel_i[2]; vel[i,3] = vel_i[3]
    end

    return pos, vel
end
```

**Purpose**  
Generates \(N\) i.i.d. initial conditions and stores them as matrices:

- `pos` is an \(N\times 3\) matrix with rows \((x_0,y_0,z_0)\).
- `vel` is an \(N\times 3\) matrix with rows \((v_x,v_y,v_z)\).

**Why matrices?**  
They are convenient for:
- passing to transport routines expecting `Nx3` inputs,
- saving to JLD2/HDF5,
- vectorized post-processing.

---

## Geometry Helpers

### `time_to_plane(pos, vel, axis, bound) -> Float64`

```julia
@inline function time_to_plane(pos::SVector{3,Float64},
                               vel::SVector{3,Float64},
                               axis::Int, bound::Float64)
    v = vel[axis]
    if v == 0.0
        return Inf
    end
    t = (bound - pos[axis]) / v
    return (t > 0.0) ? t : Inf
end
```

**Purpose**  
Computes the time for a ballistic particle to reach a coordinate plane.

**Mathematics**  
Let `axis` correspond to \(k\in\{x,y,z\}\). Solve:

\[
r_{0,k} + v_k t = b
\quad\Rightarrow\quad
t = \frac{b-r_{0,k}}{v_k}.
\]

**Return value**
- Returns \(t\) if \(t>0\).
- Returns `Inf` if:
  - \(v_k=0\) (never reaches the plane), or
  - \(t\le 0\) (intersection is behind the particle).

---

### `next_hit(pos, vel, xmin, xmax, L, zmin, zmax) -> (tmin, hit_axis, hit_side)`

```julia
function next_hit(pos::SVector{3,Float64}, vel::SVector{3,Float64},
                  xmin::Float64, xmax::Float64,
                  L::Float64,
                  zmin::Float64, zmax::Float64)
    ...
    return tmin, hit_axis, hit_side
end
```

**Purpose**  
Determines which cavity wall the particle hits next and how long it takes.

**What it does**  
It evaluates intersection times with all six planes of the rectangular cavity:

\[
x=x_{\min},\quad x=x_{\max},\quad
y=0,\quad y=L,\quad
z=z_{\min},\quad z=z_{\max}.
\]

The next event corresponds to the smallest positive time:

\[
t_{\min} = \min\{t_{x_{\min}}, t_{x_{\max}}, t_{y=0}, t_{y=L}, t_{z_{\min}}, t_{z_{\max}}\}\quad\text{with }t>0.
\]

**Returns**
\((t_{\min}, \mathrm{hit\_axis}, \mathrm{hit\_side})\) with:

- `tmin`: time until the next boundary interaction
- `hit_axis`:
  - `1` = x-wall
  - `2` = y-wall
  - `3` = z-wall
- `hit_side`:
  - `0` = lower boundary (\(x_{\min}\), \(y=0\), \(z_{\min}\))
  - `1` = upper boundary (\(x_{\max}\), \(y=L\), \(z_{\max}\))

**Notes**
- When called with `pos` inside the cavity, taking the smallest positive intersection time yields the correct next boundary without additional “inside face” checks.

---

## Lambertian Scattering

### `sample_lambertian_dir(rng, n) -> SVector{3,Float64}`

```julia
function sample_lambertian_dir(rng::AbstractRNG, n::SVector{3,Float64})
    nn = n / norm(n)
    ...
    return x*t1 + y*t2 + z*nn
end
```

**Purpose**  
Samples a random **unit direction** in the hemisphere pointed by the normal vector \(\mathbf{n}\), with cosine-weighting (Lambertian diffuse reflection).

**Target distribution**  
Let \(\theta\) be the angle to the normal. The distribution is:

\[
p(\theta,\phi) = \frac{\cos\theta}{\pi},\qquad 0\le\theta\le\frac{\pi}{2},\quad 0\le\phi<2\pi.
\]

**Sampling method (cosine-weighted hemisphere)**  
Draw \(u_1,u_2\sim \mathrm{Uniform}(0,1)\), then in a local frame where the normal is the local \(+z\) axis:

\[
r=\sqrt{u_1},\quad \phi=2\pi u_2,\quad
(x,y,z) = \big(r\cos\phi,\; r\sin\phi,\; \sqrt{1-u_1}\big).
\]

Finally rotate \((x,y,z)\) into global coordinates using an orthonormal basis \((\mathbf{t}_1,\mathbf{t}_2,\hat{\mathbf{n}})\).

---

### `inward_normal(hit_axis, hit_side) -> SVector{3,Float64}`

```julia
@inline function inward_normal(hit_axis::Int, hit_side::Int)
    ...
end
```

**Purpose**  
Returns the inward unit normal of the wall hit (so scattering is directed into the cavity).

**Meaning**
- If `hit_axis==1` and `hit_side==0`, wall is \(x=x_{\min}\) and inward normal is \(+\hat{x}\).
- If `hit_axis==1` and `hit_side==1`, wall is \(x=x_{\max}\) and inward normal is \(-\hat{x}\).
- Similarly for z walls, returning \(\pm\hat{z}\).
- y planes are included for completeness.

---

## Transport Models

### `simulate_cavity_centered(pos0, vel0, L; ...) -> NamedTuple`

**Purpose**  
Propagates particles through the cavity with Lambertian scattering on x/z walls and open y boundaries.

**Algorithm (per particle)**
1. Compute next boundary via `next_hit`.
2. Advance to the hit point.
3. If `hit_axis==2` (a y-plane):
   - `hit_side==1` \(\Rightarrow y=L\): mark `exited=true`
   - `hit_side==0` \(\Rightarrow y=0\): mark `backscatter=true`
   - record exit state and stop.
4. If `hit_axis` is x or z:
   - increment hit counters,
   - sample a Lambertian direction into the cavity,
   - keep speed constant,
   - continue.

**Returned fields**  
- `exited`: reached \(y=L\) (transmitted)
- `backscatter`: reached \(y=0\) (returned to oven)
- `exit_pos`: termination position \((x_f,y_f,z_f)\)
- `exit_vel`: termination velocity
- `exit_t`: time inside cavity
- `nhit_x`, `nhit_z`, `nhit_total`: wall collision counts
- `path_len`: total traveled distance \(\sum |\mathbf{v}|\,\Delta t\)
- `n_bounces`: scattering event counter (matches `nhit_total` here)

---

### `simulate_cavity_ballistic(pos0, vel0, L; ...) -> NamedTuple`

**Purpose**  
Ballistic baseline (geometric acceptance): if a particle hits an x/z wall before reaching \(y=L\), it is marked `lost`.

**Returned fields**
- `exited`: reached \(y=L\)
- `backscatter`: reached \(y=0\)
- `lost`: hit x/z before reaching \(y=L\)
- `exit_pos`, `exit_vel`, `exit_t`, `path_len`

---

## Analysis Utilities

### `exit_angles(out; transmitted_only=true) -> NamedTuple`

**Purpose**  
Computes angular coordinates from exit velocities.

Given \(\mathbf{v}=(v_x,v_y,v_z)\) with \(v_y>0\):

Small-angle style divergences:
\[
\theta_x = \arctan\!\left(\frac{v_x}{v_y}\right),\qquad
\theta_z = \arctan\!\left(\frac{v_z}{v_y}\right).
\]

Polar angle w.r.t. \(+y\):
\[
\theta = \arccos\!\left(\frac{v_y}{|\mathbf{v}|}\right),\qquad
\phi = \mathrm{atan2}(v_z,v_x).
\]

Returns arrays `θx`, `θz`, `θ`, `ϕ`, and `speed` for the selected particles.

---

### `transmission(out) -> Float64`

\[
T = \frac{N_{\mathrm{exited}}}{N}.
\]

---

### `compare_ballistic_diffusive(out_ball, out_diff) -> NamedTuple`

Returns:
- \(T_\mathrm{ballistic}\)
- \(T_\mathrm{diffusive}\)
- Clausing factor (as used in this workflow):
\[
K = \frac{T_\mathrm{diffusive}}{T_\mathrm{ballistic}}.
\]

---

### `solid_angle_cone(θmax) -> Float64`

Cone of half-angle \(\theta_{\max}\):
\[
\Omega(\theta_{\max}) = 2\pi\left(1-\cos\theta_{\max}\right).
\]

---

### `brightness_proxy(out; x_furnace, z_furnace, θmax) -> NamedTuple`

**Purpose**  
Computes a simple forward “brightness proxy” based on the fraction of launched particles that exit within an acceptance cone.

Let \(N_{\mathrm{in\,cone}}\) be the number of launched particles that both:
- exit at \(y=L\), and
- satisfy \(\theta \le \theta_{\max}\).

Then:
\[
B \propto \frac{(N_{\mathrm{in\,cone}}/N)}{A\,\Omega(\theta_{\max})},
\qquad A=x_\mathrm{furnace}z_\mathrm{furnace}.
\]

Returns `B`, `Nincone`, `Ω`, `A`.

---

### `brightness_reduction(out_ball, out_diff; ...) -> NamedTuple`

Returns `B_ballistic`, `B_diffusive`, and the ratio:
\[
\mathrm{reduction} = \frac{B_\mathrm{diffusive}}{B_\mathrm{ballistic}}.
\]

---

# Typical Workflow

1) Sample initial conditions at the slit \(y=0\):
```julia
pos0, vel0 = sample_initial_conditions(N, rng, p)
```

2) Run the diffusive (Lambertian) cavity model:
```julia
out_diff = simulate_cavity_centered(pos0, vel0, L;
    x_furnace=x_furnace,
    z_furnace=z_furnace)
```

3) Run ballistic baseline (geometric acceptance):
```julia
out_ball = simulate_cavity_ballistic(pos0, vel0, L;
    x_furnace=x_furnace,
    z_furnace=z_furnace)
```

4) Compare transmission and Clausing factor:
```julia
cmp = compare_ballistic_diffusive(out_ball, out_diff)
```

5) Extract angular distributions at \(y=L\):
```julia
ang = exit_angles(out_diff)
# ang.θx, ang.θz, ang.θ, ang.ϕ
```

6) Compute brightness proxy in a cone \(\theta_{\max}\):
```julia
bright = brightness_reduction(out_ball, out_diff;
    x_furnace=x_furnace,
    z_furnace=z_furnace,
    θmax=5e-3)
```

---

# Numerical Notes

- Increase `N` (e.g. \(10^5\)–\(10^7\)) for stable angular statistics.
- Ensure `max_bounces` is large enough that very long-lived trajectories do not get truncated.
- `eps_push` should be tiny compared to geometric scales; it exists only to prevent floating-point “sticking” to a wall.
- If you use extremely narrow slits, expect large `nhit_z` and broad exit angular distributions.

---
