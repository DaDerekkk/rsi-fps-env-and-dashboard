"""Player-vs-player collision on the GPU: batched port of fpssim.arena.Arena._resolve_players (cs2-web
resolveAllPlayerCollisions), see docs/gpu_move.md.

    from gpusim.collide import resolve_players
    resolve_players(state, locked, world, wall_check=None, swept=False)    # in place, once per tick right after GpuMover.step()

state  dict of tensors shaped [E, N] (N <= 16 players, 10 in the sim), platform world coordinates (z up):
       x y z vx vy vz yaw (float64), crouch grounded alive (bool).  Updated in place: x y z vx vy vz grounded.
locked [E, N] bool (or None): player is planting/defusing (Arena: p.objective is not None); locked players are not pushed.
world  the GpuMover of the map (supplies the triangle mesh for the wall check).
wall_check  after a push-out the player is moved only if the new position does not put the hull [x+-0.3, y+-0.3,
       z+0.46 .. z+height] into world geometry (otherwise that player's share of the push is dropped, the velocity exchange
       still happens).  The CPU Arena does this through BoxWorld.blockers, which exists for the whitebox maps only; with the
       Dust2 MeshWorld it has no `blockers`, so there the push is always applied.  wall_check=None keeps that behaviour
       (True on whitebox maps, False on Dust2/Mirage); pass True/False to force it.  The check is a SAT overlap test of the
       box against the collision triangles (the same test the mover uses), shrunk by BoxWorld.EPS = 1e-6 on every side.
swept  (default False = what the CPU does) True: a push longer than 0.3 m is also checked with the hulls 1/3 and 2/3 of the way.
       BoxWorld's solid boxes make the new hull alone enough; a triangle surface has no inside, so a push of more than half the hull
       width (a locked partner takes none of it: the other player gets the whole overlap, up to 0.7 m) can carry the hull through a
       wall face into a solid box unnoticed, and the mover cannot get a player out of a solid.  swept=True prevents that, at the price of
       also blocking some pushes the CPU allows (a long push that clips a box corner on the way): 0.2 to 0.35 % more crowded scenes differ.
       (cs2-web itself does not revert pushes: it re-runs the world collision on the moved players; the CPU Arena approximates that.)

Algorithm (identical order of operations, one thread per match): two passes over all pairs (i<j) of alive players in seat order;
for a pair closer than 0.70 m horizontally: if one stands on the other's head (feet within 0.22 m below the other's top and
not rising) it is lifted onto the head (grounded, vz clamped >= 0) and the pair is done; pairs with <= 0.05 m vertical overlap
and pairs of two locked players are skipped; otherwise both are pushed apart radially to 0.70 m (all of it onto the unlocked
one when only one is locked), then the approach velocity along the contact normal is cancelled by the push shares.
"""
import torch
from . import move as GM


def resolve_players(state, locked, world, wall_check=None, swept=False):
    x = state['x']
    assert x.dim() == 2, 'state tensors must be [E, N]'
    E, NP = x.shape
    dv = x.device
    if wall_check is None: wall_check = world.wall_check_default
    if locked is None: lk = torch.zeros(x.shape, dtype=torch.bool, device=dv)
    else: lk = torch.as_tensor(locked, device=dv).to(torch.bool).expand(x.shape).contiguous()
    st = []
    for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw'):
        t = state[k]; assert t.dtype == torch.float64 and t.is_contiguous(), k; st.append(t)
    for k in ('grounded', 'crouch', 'alive'):
        t = state[k]; assert t.dtype == torch.bool and t.is_contiguous(), k; st.append(t)
    st.append(lk)
    if world._cbuf.numel() < world.kmax * E:
        world._cbuf = torch.empty(world.kmax * E, dtype=torch.int32, device=dv)
    GM._ext().resolve(world.wt, world.geo, st, world._cbuf, world.err, E, NP, bool(wall_check), bool(swept), world.kmax)
    return state
