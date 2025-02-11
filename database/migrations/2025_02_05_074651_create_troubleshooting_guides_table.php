<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up()
    {
        Schema::create('troubleshooting_guides', function (Blueprint $table) {
            $table->id();
            $table->string('issue');
            $table->text('solution');
            $table->json('embedding')->nullable(); // Changed to JSON type
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('troubleshooting_guides');
    }
};