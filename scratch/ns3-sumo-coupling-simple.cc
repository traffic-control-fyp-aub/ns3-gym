/**
    EECE 502 
    ns3-sumo-coupling-simple.cc
    Purpose: Responsible for coupling ns-3 and SUMO.

    @author Rayyan Nasr 
    @author Jihad Eddine Al-Khurfan
    @version 1.0 3/12/20
*/

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traci-applications-module.h"
#include "ns3/network-module.h"
#include "ns3/traci-module.h"
#include "ns3/wave-module.h"
#include "ns3/ocb-wifi-mac.h"
#include "ns3/wifi-80211p-helper.h"
#include "ns3/wave-mac-helper.h"
#include "ns3/netanim-module.h"
#include <functional>
#include <stdlib.h>
#include <cmath>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE ("ns3-sumo-coupling-simple");

/**
    Controls operation of the program.

    In ns3, a node pool and counter which are large enough to cover all sumo vehicles, are created, according to the user's desired scenario.
    A channel and MAC are then created and set up, after which the Internet layer stack that includes routing is added.
    An IP address is assigned to each device, after which the mobility and position node pool is set up.
    Traci is then set up and SUMO is started according to the corresponding scenario, which was previously chosen by the user.
    Applications for the RSU nodes are created, and the position of each node is set.
    The interface and applications for the dynamic nodes are also created and set up after that.
    The simulation and animation are then started.
*/
int main (int argc, char *argv[])
{
  LogComponentEnable ("TraciClient", LOG_LEVEL_INFO);
  LogComponentEnable ("TrafficControlApplication", LOG_LEVEL_INFO);

  /*** 0. Command Options ***/
  uint32_t scenario = 1;
  CommandLine cmd;
  cmd.AddValue ("scenario", "simulation scenario", scenario);
  cmd.Parse (argc, argv);

  /*** 1. Create node pool and counter; large enough to cover all sumo vehicles ***/
  ns3::Time simulationTime (ns3::Seconds (30000));
  NodeContainer nodePool;

  switch (scenario)
    {
    case 0: // OSM
      nodePool.Create (2000);
      break;
    case 1: // 1-lane scenario
      nodePool.Create (2000);
      break;
    case 2: // 2-lane scenario
      nodePool.Create (2000);
      break;
    case 3: // Square scenario
    default:
      nodePool.Create (26);
      break;
    }

  uint32_t nodeCounter (0);

  /*** 2. Create and setup channel ***/
  std::string phyMode (
      "OfdmRate6MbpsBW10MHz"); // Transmission range between (between 300-400) check http://www.cse.chalmers.se/~chrpro/VANET.pdf fot specifications on transmission range
  YansWifiPhyHelper wifiPhy = YansWifiPhyHelper::Default ();
  wifiPhy.Set ("TxPowerStart", DoubleValue (20));
  wifiPhy.Set ("TxPowerEnd", DoubleValue (20));
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
  Ptr<YansWifiChannel> channel = wifiChannel.Create ();
  wifiPhy.SetChannel (channel);

  /*** 3. Create and setup MAC ***/
  wifiPhy.SetPcapDataLinkType (YansWifiPhyHelper::DLT_IEEE802_11);
  NqosWaveMacHelper wifi80211pMac = NqosWaveMacHelper::Default ();
  Wifi80211pHelper wifi80211p = Wifi80211pHelper::Default ();
  wifi80211p.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode",
                                      StringValue (phyMode), "ControlMode", StringValue (phyMode));
  NetDeviceContainer netDevices = wifi80211p.Install (wifiPhy, wifi80211pMac, nodePool);

  /*** 4. Add Internet layers stack and routing ***/
  InternetStackHelper stack;
  stack.Install (nodePool);

  /*** 5. Assign IP address to each device ***/
  Ipv4AddressHelper address;
  address.SetBase ("10.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer ipv4Interfaces;
  ipv4Interfaces = address.Assign (netDevices);

  /*** 6. Setup Mobility and position node pool ***/
  MobilityHelper mobility;
  Ptr<UniformDiscPositionAllocator> positionAlloc = CreateObject<UniformDiscPositionAllocator> ();
  positionAlloc->SetX (0.0);
  positionAlloc->SetY (320.0);
  positionAlloc->SetRho (25.0);
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (nodePool);

  /*** 7. Setup Traci and start SUMO ***/
  Ptr<TraciClient> sumoClient = CreateObject<TraciClient> ();

  switch (scenario)
    {
    case 0: // osm
      sumoClient->SetAttribute ("SumoConfigPath",
                                StringValue ("rl_fyp/sumo_files/aub-seaside/"
                                             "osm.sumocfg"));
      break;
    case 1: // 1-lane scenario
      sumoClient->SetAttribute ("SumoConfigPath",
                                StringValue ("rl_fyp/sumo_files/sumo_one_lane_highway/"
                                             "one_lane_highway.sumo.cfg"));
      break;
    case 2: // 2-lane scenario
      sumoClient->SetAttribute ("SumoConfigPath",
                                StringValue ("rl_fyp/sumo_files/sumo_two_lane_highway/"
                                             "two_lane_highway.sumo.cfg"));
      break;
    case 3: // Square scenario
    default:
      sumoClient->SetAttribute (
          "SumoConfigPath", StringValue ("rl_fyp/sumo_files/training_loop/training-loop.sumo.cfg"));
      break;
    }

  sumoClient->SetAttribute ("SumoBinaryPath", StringValue ("")); // use system installation of sumo
  sumoClient->SetAttribute ("SynchInterval", TimeValue (Seconds (0.1)));
  sumoClient->SetAttribute ("StartTime", TimeValue (Seconds (0.0)));
  sumoClient->SetAttribute ("SumoGUI", BooleanValue (true));
  sumoClient->SetAttribute ("SumoPort", UintegerValue (3400));
  sumoClient->SetAttribute ("PenetrationRate",
                            DoubleValue (1.0)); // portion of vehicles equipped with wifi
  sumoClient->SetAttribute ("SumoLogFile", BooleanValue (true));
  sumoClient->SetAttribute ("SumoStepLog", BooleanValue (false));
  sumoClient->SetAttribute ("SumoSeed", IntegerValue (10));
  sumoClient->SetAttribute ("SumoAdditionalCmdOptions", StringValue ("--verbose true"));
  sumoClient->SetAttribute ("SumoWaitForSocket", TimeValue (Seconds (1.0)));

  /*** 8. Create and Setup Applications for the RSU node and set position ***/
  Ptr<OpenGymInterface> openGymInterface;
  openGymInterface = OpenGymInterface::Get (5555);

  // ##################################################################################################3
  RsuSpeedControlHelper rsuSpeedControlHelper1 (9); // Port #9
  // rsuSpeedControlHelper1.SetAttribute ("Interval", TimeValue (Seconds (5.0))); // packet interval
  rsuSpeedControlHelper1.SetAttribute (
      "Client",
      (PointerValue) (sumoClient)); // pass TraciClient object for accessing sumo in application

  ApplicationContainer rsuSpeedControlApps = rsuSpeedControlHelper1.Install (nodePool.Get (0));
  rsuSpeedControlApps.Start (Seconds (0.0));
  rsuSpeedControlApps.Stop (simulationTime);

  Ptr<MobilityModel> mobilityRsuNode1 = nodePool.Get (0)->GetObject<MobilityModel> ();
  switch (scenario)
    {
    case 0: // osm
      mobilityRsuNode1->SetPosition (Vector (1600.0, 700.0, 3.0)); // set RSU to fixed position
      break;
    case 1: // 1-lane scenario
      mobilityRsuNode1->SetPosition (Vector (100.0, 20.0, 3.0)); // set RSU to fixed position
      break;
    case 2: // 2-lane scenario
      mobilityRsuNode1->SetPosition (Vector (100.0, 20.0, 3.0)); // set RSU to fixed position
      break;
    case 3: // Square scenario
    default:
      mobilityRsuNode1->SetPosition (Vector (100.0, 100.0, 3.0)); // set RSU to fixed position
      break;
    }
  nodeCounter++; // one node (RSU) consumed from "node pool"

  /*** 9. Setup interface and application for dynamic nodes ***/
  VehicleSpeedControlHelper vehicleSpeedControlHelper (9);
  vehicleSpeedControlHelper.SetAttribute (
      "Client",
      (PointerValue) sumoClient); // pass TraciClient object for accessing sumo in application

  // callback function for node creation
  std::function<Ptr<Node> ()> setupNewWifiNode = [&] () -> Ptr<Node> {
    if (nodeCounter >= nodePool.GetN ())
      {
        // NS_FATAL_ERROR ("Node Pool empty!: " << nodeCounter << " nodes created.");
        nodeCounter = 1;
      }

    // don't create and install the protocol stack of the node at simulation time -> take from "node pool"
    Ptr<Node> includedNode = nodePool.Get (nodeCounter);

    // Install Application
    ApplicationContainer vehicleSpeedControlApps = vehicleSpeedControlHelper.Install (includedNode);

    srand (std::time (0));
    vehicleSpeedControlApps.Start (Seconds (0.00001 * (rand () % 100000)));
    vehicleSpeedControlApps.Stop (simulationTime);
    nodeCounter++; // increment counter for next node

    return includedNode;
  };

  // callback function for node shutdown
  std::function<void (Ptr<Node>)> shutdownWifiNode = [] (Ptr<Node> exNode) {
    // stop all applications
    Ptr<VehicleSpeedControl> vehicleSpeedControl =
        exNode->GetApplication (0)->GetObject<VehicleSpeedControl> ();
    if (vehicleSpeedControl)
      {
        vehicleSpeedControl->StopApplicationNow ();
      }

    // set position outside communication range
    Ptr<ConstantPositionMobilityModel> mob = exNode->GetObject<ConstantPositionMobilityModel> ();
    mob->SetPosition (Vector (320.0 + (rand () % 25), 320.0 + (rand () % 25),
                              0.0)); // rand() for visualization purposes

    // NOTE: further actions could be required for a save shut down!
  };

  // start traci client with given function pointers
  sumoClient->SumoSetup (setupNewWifiNode, shutdownWifiNode);

  /*** 10. Setup and Start Simulation + Animation ***/
  AnimationInterface anim ("rl_fyp/netanim/ns3-sumo-coupling.xml"); // Mandatory

  Simulator::Stop (simulationTime);

  Simulator::Run ();
  openGymInterface->NotifySimulationEnd ();
  // Simulator::Destroy ();

  return 0;
}
