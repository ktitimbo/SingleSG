# This list has been updated according to the most recent values reported in NIST, CODATA2018, and IAEA 
# (see Wolfram notebook for references and uncertainties)
module AtomicSpecies
    function atoms(spec::String)
        mp  = 1.67262192595e-27 ;   # proton mass (kg)
        me  = 9.1093837139e-31 ;    # electron mass (kg) 
        μN  = 5.0507837393e-27 ;    # Nuclear magneton (J/T). RSU = 3.1e-10 NIST
        ħ   = 6.62607015e-34/2/pi ; # Reduced Planck constant (J s)
        μB  = 9.2740100657e-24 ;    # Bohr magneton (J/T). RSU = 3.0e-10 
        clight  = 299792458 ;       # Speed of light (m/s)
        mass_au = 1.66053906892e-27;    # kg/au for values check : https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
        # g-factors are given with respect to the Bohr magneton following Arimondo's work
        if spec == "39K"
            R       = 275e-12 ;         # van der Waals atomic radius (m) [https://periodic.lanl.gov/19.shtml]
            μn      = 0.3914699*μN ;    # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 3/2 ;             # Nuclear spin
            γn      = μn/ħ/spin_I ;     # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
            gfactor = μn/μB/spin_I ;    # g-factor [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]
            ahfs    = 230.8598601e6 ;   # [Hz] Properties of Potassium by T.G. Tiecke
            mass    = 38.96370668 * mass_au; # [kg] Properties of Potassium by T.G. Tiecke
        elseif spec == "41K"
            R       = 275e-12 ;         # van der Waals atomic radius (m) [https://periodic.lanl.gov/19.shtml]
            μn      = 0.2148722*μN ;    # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 3/2 ;             # Nuclear spin
            γn      = μn/ħ/spin_I ;     # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) Bruker Almanac 2012
            gfactor = μn/μB/spin_I ;    # g-factor [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]
            ahfs    = 127.0069352e6 ;   # [Hz] Properties of Potassium by T.G. Tiecke
            mass    = 40.96182576 * mass_au; # [kg] Properties of Potassium by T.G. Tiecke    
        elseif spec == "1H"
            R       = 120e-12 ;         # van der Waals atomic radius for Lithium (m) [https://periodic.lanl.gov/1.shtml]
            μn      = 2.792847351*μN ;  # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 1/2 ;             # Nuclear spin Lithium-6
            γn      = μn/ħ/spin_I ;     # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) [http://triton.iqfr.csic.es/guide/eNMR/chem/]
            gfactor = μn/μB/spin_I ;    # g-factor 1H 
            ahfs    = 1420405751.768    # [Hz] IEEE TRANSACTIONS ON INSTRUMENTATION AND MEASUREMENT, VOL. IM-19, NO. 4, NOVEMBER 1970
            mass    = 1.007825031989 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_hydrogen
        elseif spec == "2H" # no updated
            R       = 120e-12 ;         # van der Waals atomic radius for Hydrogen-2 (m) [https://periodic.lanl.gov/1.shtml]
            μn      = 0.857438231*μN ;  # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 1 ;               # Nuclear spin Hydrogen-2
            γn      = μn/ħ/spin_I       # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) Hydrogen-2 [http://triton.iqfr.csic.es/guide/eNMR/chem/]
            gfactor = μn/μB/spin_I ;    # g-factor 1H 
            ahfs    = 327.384349e6      # [Hz] Phys. Rev. 120, 1279
            mass    = 2.014101777844 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_hydrogen
        elseif spec == "6Li"
            R       = 182e-12 ;         # van der Waals atomic radius for Lithium (m) [https://periodic.lanl.gov/3.shtml]
            μn      = 0.8220428*μN ;    # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 1 ;               # Nuclear spin Lithium-6
            γn      = μn/ħ/spin_I ;     # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) Lithium-6 [http://triton.iqfr.csic.es/guide/eNMR/chem/]
            gfactor = μn/μB/spin_I ;    # g-factor 6Li [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]
            ahfs    = 152.1368407e6 ;   # [Hz] Properties of Lithium by T.G. Tiecke
            mass    = 6.0151228874 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_lithium
        elseif spec == "7Li"
            R       = 182e-12 ;         # van der Waals atomic radius for Lithium (m) [https://periodic.lanl.gov/3.shtml]
            μn      = 3.256407*μN ;     # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 3/2 ;             # Nuclear spin Lithium-7
            γn      = μn/ħ/spin_I ;     # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) [DOI: 10.1039/D0AN02088E]
            gfactor = μn/μB/spin_I ;
            ahfs    = 401.7520433e6 ;   # [Hz] PhysRevLett.111.243001
            mass    = 7.016003434 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_lithium
        elseif spec == "23Na"
            R       = 227e-12 ;         # van der Waals atomic radius for Sodium (m) [https://periodic.lanl.gov/11.shtml]
            μn      = 2.2174982*μN ;    # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            spin_I  = 3/2 ;             # Nuclear spin Sodium-23
            γn      = μn/ħ/spin_I ;     # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts)) [wikipedia or computed via g factor]
            gfactor = μn/μB/spin_I ;    # nuclear g factor https://steck.us/alkalidata/sodiumnumbers.pdf and Arimondo
            ahfs    = 885.8130644e6 ;   # [Hz] Properties of Lithium by Daniel Steck and Arimondo
            mass    = 22.9897692820 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_sodium
        elseif spec == "87Rb"
            R       = 303e-12 ;         # van der Waals atomic radius for Rubidium (m) https://periodic.lanl.gov/37.shtml
            μn      = 2.7512920*μN ;    # nuclear magnetic moment (J/T) [https://www-nds.iaea.org/nuclearmoments/]
            spin_I  = 3/2 ;             # Nuclear spin [https://www-nds.iaea.org/nuclearmoments/]
            γn      = μn/ħ/spin_I ;     # nuclear gyromagnetic ratio (rad/s/T) Bruker Almanac 2012
            gfactor = μn/μB/spin_I ;    # g-factor 87Rb [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]
            ahfs    = 3417.3413054521548e6 ;  # [Hz] Europhysics Letters 45, 558 (1999)
            mass    = 86.909180529 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_rubidium
        elseif spec == "85Rb"
            R       = 303e-12 ;         # van der Waals atomic radius for Rubidium (m) https://periodic.lanl.gov/37.shtml
            μn      = 1.3530562*μN ;    # nuclear magnetic moment (J/T) [https://www-nds.iaea.org/nuclearmoments/]
            spin_I  = 5/2 ;             # Nuclear spin [https://www-nds.iaea.org/nuclearmoments/]
            γn      = μn/ħ/spin_I  ;    # nuclear gyromagnetic ratio (rad/s/T) Bruker Almanac 2012
            gfactor = μn/μB/spin_I ;    # g-factor 87Rb [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]
            ahfs    = 1011.9108149406e6 ;      # [Hz] J. Phys. Chem. Ref. Data 51, 043102 (2022);
            mass    = 84.9117897360 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_rubidium
        elseif spec=="107Ag"
            R       = 172e-12;          # van der Waals atomic radius for Silver (m) [https://periodic.lanl.gov/47.shtml]
            μn      = 0.11352*μN;       # Nuclear magnetic moment (J/T)  [https://www-nds.iaea.org/nuclearmoments/][https://www-nds.iaea.org/publications/indc/indc-nds-0794/]
            γn      = -1.08718e7;       # Gyromagnetic ratio [gfactor*μB/ħ] (1/(Ts))
            spin_I  = 1/2 ;             # Nuclear spin Silver-107
            gfactor = γn*ħ/μB ;
            ahfs    = 1.71256e9         # [Hz] https://doi.org/10.1103/PhysRev.92.641
            mass    = 106.9050915 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_silver
        elseif spec=="133Cs"
            R       = 343e-12;          # van der Waals atomic radius for Silver (m) [https://periodic.lanl.gov/55.shtml]
            μn      = 2.5778*μN;      # Nuclear magnetic moment (J/T)
            spin_I  = 7/2 ;             # Nuclear spin Cesium-133
            γn      = μn/ħ/spin_I;       # nuclear gyromagnetic ratio (rad/s/T) Bruker Almanac 2012
            gfactor = μn/μB/spin_I ;   # g-factor 133Cs [E. Arimondo, M. Inguscio, and P. Violino. Experimental determinations of the hyperfine structure in the alkali atoms. Rev. Mod. Phys., 49(1):31, January 1977.]
            ahfs    = 2298.157946e6     # [Hz] J. Phys. Chem. Ref. Data 51, 043102 (2022);
            mass    = 132.905451958 * mass_au; # [kg] https://en.wikipedia.org/wiki/Isotopes_of_silver
        else    
            println(uppercase("This atomic species has not been introduced in our database"))
            exit(86)
        end
        println("\n\t\t USING $(spec) ATOMS")
        return [R, μn, γn, spin_I, gfactor, ahfs, mass]
    end
end