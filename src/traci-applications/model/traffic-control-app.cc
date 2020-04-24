/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2007 University of Washington
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Tempe Place, Suite 330, Boston, MA  02111-1307  USA
 */
#include "ns3/log.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-address.h"
#include "ns3/address-utils.h"
#include "ns3/nstime.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/socket.h"
#include "ns3/udp-socket.h"
#include "ns3/simulator.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include <string>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp>
#include <bits/stl_map.h> // Include for boost::split

#include "ns3/pointer.h"
#include "ns3/trace-source-accessor.h"

#include "traffic-control-app.h"

namespace ns3 {

std::vector<std::string>
split (const std::string &s, const std::string &token)
{

  std::vector<std::string> words;
  boost::split (words, s, boost::is_any_of (token), boost::token_compress_on);
  return words;
}

NS_LOG_COMPONENT_DEFINE ("TrafficControlApplication");

NS_OBJECT_ENSURE_REGISTERED (RsuSpeedControl);
NS_OBJECT_ENSURE_REGISTERED (VehicleSpeedControl);

TypeId
RsuSpeedControl::GetTypeId (void)
{
  static TypeId tid =
      TypeId ("ns3::TrafficInfoServer")
          .SetParent<Application> ()
          .SetGroupName ("Applications")
          .AddConstructor<RsuSpeedControl> ()
          .AddAttribute ("Port", "Port on which we send packets.", UintegerValue (9),
                         MakeUintegerAccessor (&RsuSpeedControl::m_port),
                         MakeUintegerChecker<uint16_t> ())
          .AddAttribute ("Interval", "The time to wait between packets", TimeValue (Seconds (2.5)),
                         MakeTimeAccessor (&RsuSpeedControl::m_interval), MakeTimeChecker ())
          .AddAttribute ("MaxPackets", "The maximum number of packets the application will send",
                         UintegerValue (100), MakeUintegerAccessor (&RsuSpeedControl::m_count),
                         MakeUintegerChecker<uint32_t> ())
          .AddAttribute ("Client", "TraCI client for SUMO", PointerValue (0),
                         MakePointerAccessor (&RsuSpeedControl::m_client),
                         MakePointerChecker<TraciClient> ())
          .AddTraceSource ("Tx", "A new packet is created and is sent",
                           MakeTraceSourceAccessor (&RsuSpeedControl::m_txTrace),
                           "ns3::Packet::TracedCallback");
  return tid;
}

RsuSpeedControl::RsuSpeedControl ()
{
  NS_LOG_FUNCTION (this);
  m_sendEvent = EventId ();
  m_port = 0;
  rx_socket = 0;
  tx_socket = 0;
  m_count = 1e9;
  m_rsu_gym_env = 0;
}

RsuSpeedControl::~RsuSpeedControl ()
{
  NS_LOG_FUNCTION (this);
  tx_socket = 0;
  rx_socket = 0;
  m_rsu_gym_env = 0;
}

void
RsuSpeedControl::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  Application::DoDispose ();
}

Ptr<RsuEnv>
RsuSpeedControl::GetEnv ()
{
  NS_LOG_FUNCTION (this);
  return m_rsu_gym_env;
}

void
RsuSpeedControl::StartApplication (void)
{
  NS_LOG_FUNCTION (this);

  // set up socket used to transmit packets
  if (tx_socket == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      tx_socket = Socket::CreateSocket (GetNode (), tid);
      Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
      Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
      Ipv4Address ipAddr = iaddr.GetBroadcast ();
      InetSocketAddress remote = InetSocketAddress (ipAddr, m_port);
      tx_socket->SetAllowBroadcast (true);
      tx_socket->Connect (remote);

      m_clear_interval = m_interval;
      // start transmitting messages after 0 seconds and update speed values after m_interval seconds
      Simulator::Schedule (m_interval, &RsuSpeedControl::ChangeSpeed, this);
      ScheduleTransmit (m_interval);
      Simulator::Schedule (m_clear_interval, &RsuSpeedControl::ClearDataTable, this);
    }

  // set up socket used to receive packets
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  rx_socket = Socket::CreateSocket (GetNode (), tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
  rx_socket->Bind (local);
  rx_socket->SetRecvCallback (MakeCallback (&RsuSpeedControl::HandleRead, this));

  // set up RSU environment
  Ptr<RsuEnv> env = CreateObject<RsuEnv> ();
  m_rsu_gym_env = env;

  NS_LOG_INFO ("New Gym Enviroment" << env << "\n");
}

void
RsuSpeedControl::ScheduleTransmit (Time dt)
{
  NS_LOG_FUNCTION (this << dt);

  // Call the send function after dt seconds
  m_sendEvent = Simulator::Schedule (dt, &RsuSpeedControl::Send, this);
}

void
RsuSpeedControl::StopApplication ()
{
  NS_LOG_FUNCTION (this);

  if (tx_socket != 0)
    {
      tx_socket->Close ();
      tx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket>> ());
    }

  if (rx_socket != 0)
    {
      rx_socket->Close ();
      rx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket>> ());
      rx_socket = 0;
    }

  Simulator::Cancel (m_sendEvent);
}

void
RsuSpeedControl::Send ()
{
  NS_LOG_FUNCTION (this << tx_socket);

  // Following is the process to broadcast list of messages to vehicles
  // ********************* Constructing message *********************

  // new string for message
  std::ostringstream msg;

  // append 0 which is used to identify an RSU
  msg << "0*";
  // Log speeds while constructing message
  NS_LOG_INFO ("\nRSU" << this->GetNode ()->GetId () << " new entries based on agent actions: \n");
  // The message will be of form: "0*id1:velocity1|id2:velocity2|...."

  std::map<std::string, vehicle_data>::iterator it = m_vehicles_data.begin ();
  while (it != m_vehicles_data.end ())
    {
      // Log new speeds
      NS_LOG_INFO ("RSU" << this->GetNode ()->GetId () << " new data = " << it->first
                         << " :: " << (it->second).velocity);

      // append vehicle id then new speed respectively
      msg << "|" << it->first << ":" << std::to_string ((it->second).velocity);
      it++;
    }
  NS_LOG_INFO ("\n");
  // terminate message by appending '\0' character
  msg << '\0';

  // *****************************************************************

  // New packet to transmit message
  Ptr<Packet> packet = Create<Packet> ((uint8_t *) msg.str ().c_str (), msg.str ().length ());

  // get ip address of current RSU for logging
  Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
  Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
  Ipv4Address ipAddr = iaddr.GetLocal ();

  // broadcast the packet
  tx_socket->Send (packet);

  NS_LOG_INFO ("0 TX ##### RSU->vehicle at time " << Simulator::Now ().GetSeconds ()
                                                  << "s - [RSU ip:" << ipAddr << "]\n");
  ScheduleTransmit (m_interval);
}

void
RsuSpeedControl::ChangeSpeed ()
{

  NS_LOG_INFO ("\nVehicles Data at RSU"
               << this->GetNode ()->GetId () << " with ip: "
               << this->GetNode ()->GetObject<Ipv4> ()->GetAddress (1, 0).GetLocal () << " \n");
  NS_LOG_INFO ("Current Entries: \n");
  // vectors to store current speed and headway data to export to the environment object
  std::vector<double> speeds;
  std::vector<double> headways;
  uint32_t i = 0;

  NS_LOG_INFO ("RSU" << this->GetNode ()->GetId ()
                     << " table at time = " << Simulator::Now ().GetSeconds () << " :\n");
  // loop over all map entries via a map iterator
  std::map<std::string, vehicle_data>::iterator it = m_vehicles_data.begin ();
  while (it != m_vehicles_data.end ())
    {

      // print the initial content in the RSU map
      NS_LOG_INFO (
          "RSU" << this->GetNode ()->GetId () << " table data = " << it->first
                << " :: " << (it->second).velocity << " :: " << (it->second).headway
                << " :: " << (it->second).fuel_consumption << " :: " << (it->second).emission_co2
                << " :: " << (it->second).emission_co << " :: " << (it->second).emission_nox
                << " :: " << (it->second).emission_pmx << " :: " << (it->second).emission_hc);

      // store speed and headway for each vehicle
      speeds.push_back ((it->second).velocity);
      headways.push_back ((it->second).headway);
      i++;
      it++;
    }

  NS_LOG_INFO ("\n");
  // call import method to send speeds and headways to environment and notify state change
  m_rsu_gym_env->ImportSpeedsAndHeadWays (headways, speeds);

  // after sending current speeds and headways, get new speeds as per RL agent actions
  std::vector<float> new_speeds = m_rsu_gym_env->ExportNewSpeeds ();

  // loop again over map entries and update speed values for each vehicle
  it = m_vehicles_data.begin ();
  i = 0;
  while (it != m_vehicles_data.end ())
    {
      // apply modulo to the new speed value increment against the actual speed 
      double new_speed_increment = fmod(static_cast<double> (new_speeds[i]),(it->second).velocity);

      // If the increment is originallly negative, let the new increment be negative
      if (static_cast<double> (new_speeds[i])<0){
          new_speed_increment= -new_speed_increment;

      }

      // add increment to current speed (notice speed will never be negative now)
      (it->second).velocity += new_speed_increment;

      i++;
      it++;
    }
  Simulator::Schedule (m_interval, &RsuSpeedControl::ChangeSpeed, this);
}

void
RsuSpeedControl::ClearDataTable ()
{
  m_vehicles_data.clear ();

  Simulator::Schedule (m_clear_interval, &RsuSpeedControl::ClearDataTable, this);
}

void
RsuSpeedControl::HandleRead (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  // receive packet from vehicle at receiving socket of RSU
  Ptr<Packet> packet;
  packet = socket->Recv ();

  // copy pakcet data into a buffer then into a string for processing
  uint8_t *buffer = new uint8_t[packet->GetSize ()];
  packet->CopyData (buffer, packet->GetSize ());
  std::string s = std::string ((char *) buffer);
  std::vector<std::string> data = split (s, "*");

  // if received data from some RSU, drop the message
  if (data[0] != "1")
    {
      return;
    }

  // save vehicle data in struct (vehicle_id, speed, headway, lane_index, emission_co2 ....)
  vehicle_data values = vehicle_data (data[1], (double) std::stod (data[2]),
                                      (double) std::stod (data[3]), (int) std::stoi (data[4]),
                                      (double) std::stod (data[5]), (double) std::stod (data[6]),
                                      (double) std::stod (data[7]), (double) std::stod (data[8]),
                                      (double) std::stod (data[9]), (double) std::stod (data[10]));

  // Inserting vehicle data to RSU database
  // using map iterator, find any matching id in the map
  std::map<std::string, vehicle_data>::iterator it = m_vehicles_data.find (values.vehicle_id);

  // if previous data of this vehicle is found, update
  if (it != m_vehicles_data.end ())
    {
      (it->second) = values;
    }
  else
    {
      if (m_vehicles_data.size () < m_rsu_gym_env->GetActionSpaceSize ())
        {
          // else insert a new entry as follows : <id>:<vehicle_data>
          m_vehicles_data.insert (std::make_pair (values.vehicle_id, values));
        }
    }

  // get ip of RSU for logging
  Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
  Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
  Ipv4Address ipAddr = iaddr.GetLocal ();

  NS_LOG_INFO ("3 RX ##### vehicle->RSU at time " << Simulator::Now ().GetSeconds ()
                                                  << "s - [RSU ip:" << ipAddr << "]"
                                                  << "[from vehicle:" << values.vehicle_id << "]"
                                                  << "[rx vel:" << values.velocity << "m/s]"
                                                  << "[rx headway:" << values.headway << "]\n");
}

TypeId
VehicleSpeedControl::GetTypeId (void)
{
  static TypeId tid =
      TypeId ("ns3::TrafficInfoClient")
          .SetParent<Application> ()
          .SetGroupName ("Applications")
          .AddConstructor<VehicleSpeedControl> ()
          .AddAttribute ("Port", "The port on which the client will listen for incoming packets.",
                         UintegerValue (0), MakeUintegerAccessor (&VehicleSpeedControl::m_port),
                         MakeUintegerChecker<uint16_t> ())
          .AddAttribute ("Interval", "The time to wait between packets", TimeValue (Seconds (2.5)),
                         MakeTimeAccessor (&VehicleSpeedControl::m_interval), MakeTimeChecker ())
          .AddAttribute ("Client", "TraCI client for SUMO", PointerValue (0),
                         MakePointerAccessor (&VehicleSpeedControl::m_client),
                         MakePointerChecker<TraciClient> ());
  return tid;
}

VehicleSpeedControl::VehicleSpeedControl ()
{
  NS_LOG_FUNCTION (this);
  m_sendEvent = EventId ();
  tx_socket = 0;
  rx_socket = 0;
  m_port = 0;
  m_client = nullptr;
  last_velocity = 0;
}

VehicleSpeedControl::~VehicleSpeedControl ()
{
  NS_LOG_FUNCTION (this);
  tx_socket = 0;
  rx_socket = 0;
}

void
VehicleSpeedControl::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  Application::DoDispose ();
}

void
VehicleSpeedControl::StartApplication (void)
{
  NS_LOG_FUNCTION (this);

  // set up socket used to receive packets
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  rx_socket = Socket::CreateSocket (GetNode (), tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
  rx_socket->Bind (local);
  rx_socket->SetRecvCallback (MakeCallback (&VehicleSpeedControl::HandleRead, this));

  // set up socket used to transmit packets
  if (tx_socket == 0)
    {
      TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
      tx_socket = Socket::CreateSocket (GetNode (), tid);
      Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
      Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
      Ipv4Address ipAddr = iaddr.GetBroadcast ();
      InetSocketAddress remote = InetSocketAddress (ipAddr, m_port);
      tx_socket->SetAllowBroadcast (true);
      tx_socket->Connect (remote);

      ScheduleTransmit (m_interval);
    }
}

void
VehicleSpeedControl::StopApplication ()
{
  NS_LOG_FUNCTION (this);

  if (tx_socket != 0)
    {
      tx_socket->Close ();
      tx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket>> ());
    }

  if (rx_socket != 0)
    {
      rx_socket->Close ();
      rx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket>> ());
      rx_socket = 0;
    }
  Simulator::Cancel (m_sendEvent);
}

void
VehicleSpeedControl::StopApplicationNow ()
{
  NS_LOG_FUNCTION (this);
  StopApplication ();
}

void
VehicleSpeedControl::HandleRead (Ptr<Socket> socket)
{
  NS_LOG_FUNCTION (this << socket);

  // receive packet from RSU
  Ptr<Packet> packet;
  packet = socket->Recv ();

  // copy packet data into a buffer then to a string to process
  uint8_t *buffer = new uint8_t[packet->GetSize ()];
  packet->CopyData (buffer, packet->GetSize ());
  std::string s = std::string ((char *) buffer);
  std::vector<std::string> data = split (s, "*");

  // if packet is received from a vehicle, dump the message
  if (data[0] != "0")
    {
      return;
    }

  // parse data received to find current vehicle id
  double velocity = -999;
  std::vector<std::string> map_data = split (data[1], "|");
  for (uint8_t i = 0; i < map_data.size (); i++)
    {
      std::vector<std::string> parameters = split (map_data[i], ":");

      // when id of current vehicle is found, get respective speed and save value
      if (parameters[0] == m_client->GetVehicleId (this->GetNode ()))
        {
          velocity = std::stod (parameters[1]);
        }
    }

  // if current id was not found in the message, discard.
  if (velocity == -999)
    {
      return;
    }

  // get ip of current vehicle for logging
  Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
  Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
  Ipv4Address ipAddr = iaddr.GetLocal ();

  NS_LOG_INFO ("1 RX ***** RSU->vehicle at time "
               << Simulator::Now ().GetSeconds () << "s - "
               << "[vehicle ip:" << ipAddr << "]"
               << "[vehicle id:" << m_client->GetVehicleId (this->GetNode ()) << "]"
               << "[vel:"
               << m_client->TraCIAPI::vehicle.getSpeed (m_client->GetVehicleId (this->GetNode ()))
               << "m/s]"
               << "[rx vel:" << velocity << "m/s]\n");

  if (velocity >= 0)
    {
      m_client->TraCIAPI::vehicle.setSpeed (m_client->GetVehicleId (this->GetNode ()), velocity);
      last_velocity = velocity;
    }
}

void
VehicleSpeedControl::Send ()
{
  NS_LOG_FUNCTION (this << tx_socket);

  // Get Headway Just before sending
  // Headway in seconds  = Headway in meters / velocity

  last_velocity = m_client->TraCIAPI::vehicle.getSpeed (m_client->GetVehicleId (this->GetNode ()));
  last_headway =
      m_client->TraCIAPI::vehicle.getLeader (m_client->GetVehicleId (this->GetNode ()), 0).second /
      last_velocity;

  if (last_velocity <= 0.0 || last_velocity == 1 / 0.0)
    {
      last_velocity = 0.0;
    }
  if (last_headway <= 0.0 || last_headway == 1 / 0.0)
    {
      last_headway = 0.0;
    }

  // get vehicle ip for logging
  Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
  Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
  Ipv4Address ipAddr = iaddr.GetLocal ();

  // ********************* Constructing message *********************

  // new message string
  std::ostringstream msg;

  // append 1 which is the identifier of a vehicle, append the current velocity and headway and other parameters
  msg << "1*" << m_client->GetVehicleId (this->GetNode ()) << "*" << std::to_string (last_velocity)
      << "*" << std::to_string (last_headway) << "*"
      << std::to_string (
             m_client->TraCIAPI::vehicle.getLaneIndex (m_client->GetVehicleId (this->GetNode ())))
      << "*"
      << std::to_string (m_client->TraCIAPI::vehicle.getFuelConsumption (
             m_client->GetVehicleId (this->GetNode ())))
      << "*"
      << std::to_string (
             m_client->TraCIAPI::vehicle.getCO2Emission (m_client->GetVehicleId (this->GetNode ())))
      << "*"
      << std::to_string (
             m_client->TraCIAPI::vehicle.getCOEmission (m_client->GetVehicleId (this->GetNode ())))
      << "*"
      << std::to_string (
             m_client->TraCIAPI::vehicle.getNOxEmission (m_client->GetVehicleId (this->GetNode ())))
      << "*"
      << std::to_string (
             m_client->TraCIAPI::vehicle.getPMxEmission (m_client->GetVehicleId (this->GetNode ())))
      << "*"
      << std::to_string (
             m_client->TraCIAPI::vehicle.getHCEmission (m_client->GetVehicleId (this->GetNode ())))
      << '\0';
  Ptr<Packet> packet = Create<Packet> ((uint8_t *) msg.str ().c_str (), msg.str ().length ());

  // send packet
  tx_socket->Send (packet);
  NS_LOG_INFO ("2 TX ***** Vehicle->RSU at time "
               << Simulator::Now ().GetSeconds () << "s - "
               << "[vehicle ip:" << ipAddr << "]"
               << "[vehicle id:" << m_client->GetVehicleId (this->GetNode ()) << "]"
               << "[tx vel:" << last_velocity << "m/s]"
               << "[tx headway:" << last_headway << "s]\n");

  ScheduleTransmit (m_interval);
}

void
VehicleSpeedControl::ScheduleTransmit (Time dt)
{
  NS_LOG_FUNCTION (this << dt);
  m_sendEvent = Simulator::Schedule (dt, &VehicleSpeedControl::Send, this);
}

} // Namespace ns3
