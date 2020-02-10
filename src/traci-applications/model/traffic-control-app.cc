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
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp>
#include <bits/stl_map.h> // Include for boost::split

#include "ns3/pointer.h"
#include "ns3/trace-source-accessor.h"

#include "traffic-control-app.h"

namespace ns3
{
	std::vector<std::string> split(const std::string& s, const std::string& token) {
		
		std::vector<std::string> words;
		boost::split(words, s, boost::is_any_of(token), boost::token_compress_on);
		return words;
	}

  NS_LOG_COMPONENT_DEFINE("TrafficControlApplication");

  NS_OBJECT_ENSURE_REGISTERED(RsuSpeedControl);
  NS_OBJECT_ENSURE_REGISTERED(VehicleSpeedControl);

  TypeId
  RsuSpeedControl::GetTypeId (void)
  {
  static TypeId tid = TypeId ("ns3::TrafficInfoServer")
    .SetParent<Application> ()
    .SetGroupName("Applications")
    .AddConstructor<RsuSpeedControl> ()
    .AddAttribute ("Port", "Port on which we send packets.",
                   UintegerValue (9),
                   MakeUintegerAccessor (&RsuSpeedControl::m_port),
                   MakeUintegerChecker<uint16_t> ())
    .AddAttribute ("Interval",
                   "The time to wait between packets",
                   TimeValue (Seconds (5.0)),
                   MakeTimeAccessor (&RsuSpeedControl::m_interval),
                   MakeTimeChecker ())
   .AddAttribute ("MaxPackets",
                      "The maximum number of packets the application will send",
                      UintegerValue (100),
                      MakeUintegerAccessor (&RsuSpeedControl::m_count),
                      MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("Client",
                   "TraCI client for SUMO",
                   PointerValue (0),
                   MakePointerAccessor (&RsuSpeedControl::m_client),
                   MakePointerChecker<TraciClient> ())
    .AddTraceSource ("Tx", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&RsuSpeedControl::m_txTrace),
                     "ns3::Packet::TracedCallback")
  ;
    return tid;
  }

  RsuSpeedControl::RsuSpeedControl ()
  {
    NS_LOG_FUNCTION(this);
    m_sendEvent = EventId ();
    m_port = 0;
    rx_socket = 0;
    tx_socket = 0;
    m_count = 1e9;
  }

  RsuSpeedControl::~RsuSpeedControl ()
  {
    NS_LOG_FUNCTION(this);
    tx_socket = 0;
    rx_socket = 0;
  }

  void
  RsuSpeedControl::DoDispose (void)
  {
    NS_LOG_FUNCTION(this);
    Application::DoDispose ();
  }

  void
  RsuSpeedControl::StartApplication (void)
  {
    NS_LOG_FUNCTION(this);

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

        ScheduleTransmit (Seconds (0.0));
        Simulator::Schedule (Seconds (5.0), &RsuSpeedControl::ChangeSpeed, this);
      }
            TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
    rx_socket = Socket::CreateSocket (GetNode (), tid);
    InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
    rx_socket->Bind (local);
    rx_socket->SetRecvCallback (MakeCallback (&RsuSpeedControl::HandleRead, this));

  }

  void
  RsuSpeedControl::ScheduleTransmit (Time dt)
  {
    NS_LOG_FUNCTION(this << dt);
    m_sendEvent = Simulator::Schedule (dt, &RsuSpeedControl::Send, this);
  }

  void
  RsuSpeedControl::StopApplication ()
  {
    NS_LOG_FUNCTION(this);

    if (tx_socket != 0)
      {
        tx_socket->Close ();
        tx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
      }
	
	if (rx_socket != 0)
      {
        rx_socket->Close ();
        rx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
        rx_socket = 0;
      }

    Simulator::Cancel (m_sendEvent);
  }

  void
  RsuSpeedControl::Send ()
  {
    NS_LOG_FUNCTION(this << tx_socket);

	// ********************* Constructing message ********************* 
    std::ostringstream msg;
    msg << "0*";
	
	// The message will be of form: "0|id1:velocity1|id2:velocity2|...."
	
	std::map<std::string, std::pair<double,double>>::iterator it = m_vehicles_data.begin();
    while(it != m_vehicles_data.end())
    {
		msg << "|" << it->first<<":"<<std::to_string((it->second).first);
        it++;
    }
	
	msg << '\0';
	
	// *****************************************************************
	
    Ptr<Packet> packet = Create<Packet> ((uint8_t*) msg.str ().c_str (), msg.str ().length ());

    Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
    Ipv4Address ipAddr = iaddr.GetLocal ();

    tx_socket->Send (packet);
    NS_LOG_INFO("TX ##### RSU->vehicle at time " << Simulator::Now().GetSeconds()
                << "s - [RSU ip:" << ipAddr << "]\n");

    ScheduleTransmit (m_interval);
  }

  void
  RsuSpeedControl::ChangeSpeed ()
  {
	
	NS_LOG_INFO("\nVehicles Data at RSU with ip: " << this->GetNode ()->GetObject<Ipv4> ()->GetAddress (1, 0).GetLocal () << " \n");

	
	std::map<std::string, std::pair<double,double>>::iterator it = m_vehicles_data.begin();
    while(it != m_vehicles_data.end())
    {
        NS_LOG_INFO(it->first<<" :: "<<(it->second).first<<" :: "<<(it->second).second);
		
		//TODO CHANGE THIS  
		(it->second).first = rand () % 60;
        it++;
    }
	
	NS_LOG_INFO("\n");
				
    Simulator::Schedule (Seconds (5.0), &RsuSpeedControl::ChangeSpeed, this);
  }

void
  RsuSpeedControl::HandleRead (Ptr<Socket> socket)
  {
    NS_LOG_FUNCTION(this << socket);
    Ptr<Packet> packet;
    packet = socket->Recv ();

    uint8_t *buffer = new uint8_t[packet->GetSize ()];
    packet->CopyData (buffer, packet->GetSize ());
    std::string s = std::string ((char*) buffer);
	std::vector<std::string> data = split (s, "*");
	
	if (data[0]!="1"){return;}
	
	std::string receivedID = data[1];
    double velocity = (double) std::stod (data[2]);
	double headway = (double) std::stod (data[3]);
	
	// Inserting vehicle data to RSU databse
	std::map<std::string, std::pair<double,double>>::iterator it = m_vehicles_data.find(receivedID); 
	if (it != m_vehicles_data.end()){
		(it->second).first = velocity;
		(it->second).second = headway;
	}else {
			m_vehicles_data.insert(std::make_pair(receivedID, std::make_pair(velocity,headway)));
	}

    Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
    Ipv4Address ipAddr = iaddr.GetLocal ();

    NS_LOG_INFO("RX ##### vehicle->RSU at time " << Simulator::Now().GetSeconds()
        << "s - [RSU ip:" << ipAddr << "]"
		<< "[from vehicle:" << receivedID << "]"
        << "[rx vel:" << velocity << "m/s]"
		<< "[rx headway:" << headway << "]\n");
  }

  TypeId
  VehicleSpeedControl::GetTypeId (void)
  {
    static TypeId tid =
        TypeId ("ns3::TrafficInfoClient")
        .SetParent<Application> ()
        .SetGroupName ("Applications")
        .AddConstructor<VehicleSpeedControl> ()
        .AddAttribute (
            "Port", "The port on which the client will listen for incoming packets.",
            UintegerValue (0),
            MakeUintegerAccessor (&VehicleSpeedControl::m_port),
            MakeUintegerChecker<uint16_t> ())
        .AddAttribute ("Interval",
                   "The time to wait between packets",
                   TimeValue (Seconds (5.0)),
                   MakeTimeAccessor (&VehicleSpeedControl::m_interval),
                   MakeTimeChecker ())
        .AddAttribute (
            "Client", "TraCI client for SUMO",
            PointerValue (0),
            MakePointerAccessor (&VehicleSpeedControl::m_client),
            MakePointerChecker<TraciClient> ());
    return tid;
  }

  VehicleSpeedControl::VehicleSpeedControl ()
  {
    NS_LOG_FUNCTION(this);
    m_sendEvent = EventId ();
	tx_socket = 0;
    rx_socket = 0;
    m_port = 0;
    m_client = nullptr;
    last_velocity = -1;
  }

  VehicleSpeedControl::~VehicleSpeedControl ()
  {
    NS_LOG_FUNCTION(this);
    tx_socket = 0;
    rx_socket = 0;
  }

  void
  VehicleSpeedControl::DoDispose (void)
  {
    NS_LOG_FUNCTION(this);
    Application::DoDispose ();
  }

  void
  VehicleSpeedControl::StartApplication (void)
  {
    NS_LOG_FUNCTION(this);

    TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
    rx_socket = Socket::CreateSocket (GetNode (), tid);
    InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
    rx_socket->Bind (local);
    rx_socket->SetRecvCallback (MakeCallback (&VehicleSpeedControl::HandleRead, this));

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

        ScheduleTransmit (Seconds (0.0));
        Simulator::Schedule (Seconds (5.0), &VehicleSpeedControl::Send, this);
      }
  }

  void
  VehicleSpeedControl::StopApplication ()
  {
    NS_LOG_FUNCTION(this);
	
	if (tx_socket != 0)
      {
        tx_socket->Close ();
        tx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
      }
	
    if (rx_socket != 0)
      {
        rx_socket->Close ();
        rx_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
        rx_socket = 0;
      }
  }

  void
  VehicleSpeedControl::StopApplicationNow ()
  {
    NS_LOG_FUNCTION(this);
    StopApplication ();
  }

  void
  VehicleSpeedControl::HandleRead (Ptr<Socket> socket)
  {
    NS_LOG_FUNCTION(this << socket);
    Ptr<Packet> packet;
    packet = socket->Recv ();

    uint8_t *buffer = new uint8_t[packet->GetSize ()];
    packet->CopyData (buffer, packet->GetSize ());
    std::string s = std::string ((char*) buffer);
	std::vector<std::string> data = split (s, "*");

	if (data[0]!="0"){
		NS_LOG_INFO("Dumping data from vehicle");
		return;
	}
	
	double velocity = -999;
	std::vector<std::string> map_data = split (data[1], "|");
	for (uint8_t i=0;i< map_data.size(); i++ ){
		std::vector<std::string> parameters = split(map_data[i],":");
		if (parameters[0] == m_client->GetVehicleId(this->GetNode())){
			velocity = std::stod(parameters[1]);
		}
	}
	
	if (velocity == -999){
		NS_LOG_INFO("Dumping data from RSU");
		return;
	}


    Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
    Ipv4Address ipAddr = iaddr.GetLocal ();
	
    NS_LOG_INFO("RX ***** RSU->vehicle at time " << Simulator::Now().GetSeconds()
        << "s - "
        << "[vehicle ip:" << ipAddr << "]"
		<< "[vehicle id:" << m_client->GetVehicleId(this->GetNode()) << "]"
        << "[vel:" << m_client->TraCIAPI::vehicle.getSpeed(m_client->GetVehicleId(this->GetNode())) << "m/s]"
        << "[rx vel:" << velocity << "m/s]\n");

    if (velocity != last_velocity)
      {
        m_client->TraCIAPI::vehicle.setSpeed (m_client->GetVehicleId (this->GetNode ()), velocity);
        last_velocity = velocity;
      }
  }
  
void
  VehicleSpeedControl::Send ()
  {
    NS_LOG_FUNCTION(this << tx_socket);
	
	// Get Headway Just before sending 
	// Headway in seconds  = Headway in meters / velocity
	if (last_velocity <= 0){
		last_headway = 0.0;
	}else{
		last_headway = m_client->TraCIAPI::vehicle.getLeader (m_client->GetVehicleId(this->GetNode()),0).second / last_velocity;
	}
	
    std::ostringstream msg;
    msg << "1*" << m_client->GetVehicleId(this->GetNode()) << "*" << std::to_string (last_velocity) << "*" << std::to_string (last_headway) <<'\0';
    Ptr<Packet> packet = Create<Packet> ((uint8_t*) msg.str ().c_str (), msg.str ().length ());

    Ptr<Ipv4> ipv4 = this->GetNode ()->GetObject<Ipv4> ();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress (1, 0);
    Ipv4Address ipAddr = iaddr.GetLocal ();

   tx_socket->Send (packet);
    NS_LOG_INFO("TX ***** Vehicle->RSU at time " << Simulator::Now().GetSeconds()
                << "s - "
			    << "[vehicle ip:" << ipAddr << "]"
				<< "[vehicle id:" << m_client->GetVehicleId(this->GetNode()) << "]"
                << "[tx vel:" << last_velocity << "m/s]"
				<< "[tx headway:" << last_headway << "s]\n");

    ScheduleTransmit (m_interval);
  }

 void
  VehicleSpeedControl::ScheduleTransmit (Time dt)
  {
    NS_LOG_FUNCTION(this << dt);
    m_sendEvent = Simulator::Schedule (dt, &VehicleSpeedControl::Send, this);
  }

} // Namespace ns3
