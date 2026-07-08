const role = (bus) => String(bus?.role ?? "").toLowerCase();

export function applyTopologyChange(settings, { buses: nextBuses, lines: nextLines }) {
  const buses = nextBuses ?? settings.buses ?? [];
  const lines = nextLines ?? settings.lines ?? [];
  const validIds = new Set(buses.map((bus) => Number(bus.id)));
  const fallbackBus = buses[0]?.id ?? 0;
  const chooseBus = (current, ...wantedRoles) => {
    if (validIds.has(Number(current))) return current;
    for (const wanted of wantedRoles) {
      const match = buses.find((bus) => role(bus) === wanted);
      if (match) return match.id;
    }
    return fallbackBus;
  };
  const cleanLines = lines.filter(
    (line) =>
      validIds.has(Number(line.from_bus)) &&
      validIds.has(Number(line.to_bus)) &&
      Number(line.from_bus) !== Number(line.to_bus)
  );

  // Buses that are new or whose role just changed: placing a power-source /
  // asset node in the designer wires the matching asset onto it.
  const prevRoleById = new Map(
    (settings.buses ?? []).map((bus) => [Number(bus.id), role(bus)])
  );
  const changed = buses.filter((bus) => prevRoleById.get(Number(bus.id)) !== role(bus));

  let pvArrays = (settings.pv_arrays ?? []).map((pv) => ({
    ...pv,
    bus: chooseBus(pv.bus, "pv"),
  }));
  let loads = (settings.loads ?? []).map((load) => ({
    ...load,
    bus: chooseBus(load.bus, "load"),
  }));
  let battery = { ...settings.battery, bus: chooseBus(settings.battery.bus, "battery") };
  let ev = { ...settings.ev, bus: chooseBus(settings.ev.bus, "ev") };
  let diesel = { ...settings.diesel, bus: chooseBus(settings.diesel.bus, "diesel", "main") };

  for (const bus of changed) {
    const id = Number(bus.id);
    switch (role(bus)) {
      case "pv":
        if (!pvArrays.some((pv) => Number(pv.bus) === id)) {
          pvArrays = [...pvArrays, { name: `PV array ${pvArrays.length}`, bus: id, p_mw: 0.05 }];
        }
        break;
      case "diesel":
        diesel = { ...diesel, bus: id, enabled: true };
        break;
      case "battery":
        battery = { ...battery, bus: id };
        break;
      case "ev":
        ev = { ...ev, bus: id };
        break;
      case "load":
        if (!loads.some((load) => Number(load.bus) === id)) {
          loads = [...loads, { name: `Load ${loads.length}`, bus: id, p_mw: 0.05, q_mvar: 0.01 }];
        }
        break;
      default:
        break; // grid buses become the slack automatically; main/bus need no asset
    }
  }

  return {
    ...settings,
    buses,
    lines: cleanLines,
    battery,
    ev,
    diesel,
    pv_arrays: pvArrays,
    loads,
    topology: {
      ...settings.topology,
      n_pv: pvArrays.length,
      n_load: loads.length,
    },
  };
}
