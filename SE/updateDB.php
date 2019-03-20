<?php require_once("ratH.html"); 
	// complete this code to list all entries 
	// in the DBtest table
	$username = "tblan424";
	$password = "drap6deqAthe";
	$DB_HOST = "localhost";
	$DB_NAME = "reg_pbp_2009.csv";
	//db_connect($username, $password);
	//global $connection;

	$connection = mysqli_connect($DB_HOST, $username, $password, $DB_NAME);  

	print("Hello");
	//session_start(); 
require_once("footerT.html"); ?>